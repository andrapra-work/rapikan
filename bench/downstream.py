"""Downstream check: does preprocessing change classification results?

Same model for every pipeline (TF-IDF 1-2 grams + logistic regression); only
the preprocessing changes. Scores are macro-F1 on the evaluation split, with a
95% paired bootstrap interval for each pipeline and for the difference against
the minimal pipeline (lowercase + strip non-alphanumerics) and the tutorial pipeline.

Datasets (download them yourself; none are bundled):
- smsa:  IndoNLU SmSA sentiment. Already lowercased and space-tokenized.
         dir with train_preprocess.tsv, valid_preprocess.tsv
- emot:  IndoNLU EmoT emotion (5 classes). Raw tweets: case, hashtags, slang.
         dir with train.csv, valid.csv (columns label, tweet)
- nusax: NusaX-senti, Indonesian split. Original casing and punctuation.
         dir with train.csv, test.csv (columns id, text, label)

Test labels of SmSA and EmoT are hidden, so their valid split is the eval set.

    python bench/downstream.py --smsa DIR --emot DIR --nusax DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from rapikan import presets

CACHE = Path.home() / ".cache" / "rapikan-bench" / "sastrawi_stems.json"


def load_smsa(d):
    def read(name):
        X, y = [], []
        for line in (Path(d) / name).read_text(encoding="utf-8").splitlines():
            t, lab = line.rsplit("\t", 1)
            X.append(t); y.append(lab)
        return X, y
    return read("train_preprocess.tsv"), read("valid_preprocess.tsv")


def load_csv(d, train, test, text_col):
    def read(name):
        rows = list(csv.DictReader(open(Path(d) / name, encoding="utf-8")))
        return [r[text_col] for r in rows], [r["label"] for r in rows]
    return read(train), read(test)


class Tutorial:
    """The pipeline most tutorials use: lowercase, regex, Sastrawi stopwords, Sastrawi stem."""

    def __init__(self):
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        self.stemmer = StemmerFactory().create_stemmer()
        self.stop = set(StopWordRemoverFactory().get_stop_words())
        self.cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    def __call__(self, text):
        out = []
        for w in re.sub(r"[^a-z0-9]+", " ", text.lower()).split():
            if w in self.stop:
                continue
            if w not in self.cache:
                self.cache[w] = self.stemmer.stem(w)
            out.append(self.cache[w])
        return out

    def save(self):
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(self.cache))


def pipelines(tutorial):
    out = {
        "none (whitespace)": str.split,
        "minimal (lower+regex)": lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).split(),
    }
    if tutorial:
        out["tutorial (Sastrawi)"] = tutorial
    for name in ("classic", "sentiment", "topic"):
        p = getattr(presets, name)()
        out[f"rapikan {name}"] = lambda t, p=p: p(t).words
    return out


def predict(prep, Xtr, ytr, Xte):
    tr = [" ".join(prep(t)) for t in Xtr]
    te = [" ".join(prep(t)) for t in Xte]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                          token_pattern=r"\S+", lowercase=False)
    clf = LogisticRegression(max_iter=3000, C=4.0)
    clf.fit(vec.fit_transform(tr), ytr)
    return list(clf.predict(vec.transform(te)))


def boot(y, preds, base, n_boot=1000, seed=0):
    """Macro-F1 CI for each pipeline, and CI of the paired difference against base pipelines."""
    rng = random.Random(seed)
    n = len(y)
    samples = {k: [] for k in preds}
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        yy = [y[i] for i in idx]
        for k, p in preds.items():
            samples[k].append(f1_score(yy, [p[i] for i in idx], average="macro"))

    def ci(v):
        v = sorted(v)
        return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]

    out = {}
    for k in preds:
        row = {"f1": f1_score(y, preds[k], average="macro"), "ci": ci(samples[k])}
        for b in base:
            if b in preds and b != k:
                d = [a - c for a, c in zip(samples[k], samples[b])]
                row[b] = (f1_score(y, preds[k], average="macro") - f1_score(y, preds[b], average="macro"), ci(d))
        out[k] = row
    return out


def report(title, data, pipes):
    (Xtr, ytr), (Xte, yte) = data
    preds = {name: predict(fn, Xtr, ytr, Xte) for name, fn in pipes.items()}
    base = ["minimal (lower+regex)", "tutorial (Sastrawi)"]
    res = boot(yte, preds, base)
    print(f"\n## {title} (train {len(Xtr)}, eval {len(Xte)}, {len(set(ytr))} classes)\n")
    print("| pipeline | macro-F1 | 95% CI | vs minimal | vs tutorial |")
    print("|---|---|---|---|---|")

    def fmt(r, b):
        if b not in r:
            return "-"
        d, (lo, hi) = r[b]
        mark = " *" if lo > 0 or hi < 0 else ""
        return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]{mark}"

    for k, r in res.items():
        lo, hi = r["ci"]
        print(f"| {k} | {r['f1']:.3f} | {lo:.3f}-{hi:.3f} | {fmt(r, base[0])} | {fmt(r, base[1])} |")
    print("\n`*` = 95% interval of the difference excludes zero.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smsa"); ap.add_argument("--emot"); ap.add_argument("--nusax")
    a = ap.parse_args()
    try:
        tut = Tutorial()
    except ImportError:
        tut = None
    pipes = pipelines(tut)
    if a.smsa:
        report("SmSA sentiment (pre-cleaned text)", load_smsa(a.smsa), pipes)
    if a.emot:
        report("EmoT emotion (raw tweets)", load_csv(a.emot, "train.csv", "valid.csv", "tweet"), pipes)
    if a.nusax:
        report("NusaX-senti Indonesian (original casing)", load_csv(a.nusax, "train.csv", "test.csv", "text"), pipes)
    if tut:
        tut.save()


if __name__ == "__main__":
    main()
