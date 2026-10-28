"""Downstream check: does preprocessing change sentiment accuracy on SmSA?

SmSA (IndoNLU, CC BY-SA 4.0): train 11,000 / valid 1,260 (test labels are hidden,
so valid is used as the test set). Model: TF-IDF (1-2 grams) + logistic regression,
same for every pipeline; only the preprocessing changes.

Note: SmSA text is already lowercased and space-tokenized, with most emoji gone,
so this measures negation, slang and stemming effects, not emoji handling.

    python bench/downstream_smsa.py path/to/smsa_dir
"""
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from rapikan import presets


def load(p):
    X, y = [], []
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        t, lab = line.rsplit("\t", 1)
        X.append(t); y.append(lab)
    return X, y


def run(name, prep, Xtr, ytr, Xte, yte):
    tr = [" ".join(prep(t)) for t in Xtr]
    te = [" ".join(prep(t)) for t in Xte]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+")
    clf = LogisticRegression(max_iter=2000, C=4.0)
    clf.fit(vec.fit_transform(tr), ytr)
    pred = clf.predict(vec.transform(te))
    print(f"| {name:<22} | {accuracy_score(yte, pred):.3f} | {f1_score(yte, pred, average='macro'):.3f} |")


def main(d):
    Xtr, ytr = load(Path(d) / "train_preprocess.tsv")
    Xte, yte = load(Path(d) / "valid_preprocess.tsv")
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        sas = StemmerFactory().create_stemmer()
        sw = set(StopWordRemoverFactory().get_stop_words())
        cache = {}
        def tutorial(t):
            out = []
            for w in t.split():
                if not w.isalpha() or w in sw:
                    continue
                if w not in cache:
                    cache[w] = sas.stem(w)
                out.append(cache[w])
            return out
    except ImportError:
        tutorial = None

    print("| pipeline               | acc   | macro-F1 |")
    print("|------------------------|-------|----------|")
    run("none (whitespace)", str.split, Xtr, ytr, Xte, yte)
    if tutorial:
        run("tutorial (Sastrawi)", tutorial, Xtr, ytr, Xte, yte)
    for name in ("classic", "sentiment", "topic"):
        p = getattr(presets, name)()
        run(f"rapikan {name}", lambda t, p=p: p(t).words, Xtr, ytr, Xte, yte)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/smsa")
