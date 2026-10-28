"""Compare rapikan with other open-source Indonesian stemmers on the same gold sets.

Stemmers (each optional; skipped if not installed):
- Sastrawi (PySastrawi): Nazief-Adriani / confix-stripping family.
- mpstemmer (github.com/ariaghora/mpstemmer): confix-stripping (Adriani et al. 2007),
  extended for nonstandard words. Python 3.11 or older.
- nlp-id Lemmatizer (Kumparan): Python <3.13.

Every stemmer sees the same lowercase word; no caching tricks, no tuning.
Accuracy has a 95% bootstrap interval over words, and McNemar's exact test
compares rapikan against each other stemmer on the same words.

    python bench/compare_stemmers.py
"""
from __future__ import annotations

import random
import sys
import time
from math import comb
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run_benchmark import load_gold  # noqa: E402

from rapikan import Stemmer  # noqa: E402

SETS = [
    ("ud_csui_stems.tsv", "UD CSUI affixed (independent)"),
    ("heldout_stems.tsv", "rapikan held-out"),
    ("gold_stems.tsv", "rapikan development (tuned on)"),
]


def stemmers():
    out = {"rapikan": Stemmer().stem}
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        out["Sastrawi"] = StemmerFactory().create_stemmer().stem
    except ImportError:
        pass
    try:
        from mpstemmer import MPStemmer
        out["mpstemmer"] = MPStemmer().stem
    except ImportError:
        pass
    try:
        from nlp_id.lemmatizer import Lemmatizer
        out["nlp-id"] = Lemmatizer().lemmatize
    except ImportError:
        pass
    return out


def boot_ci(correct, n_boot=2000, seed=0):
    rng = random.Random(seed)
    n = len(correct)
    accs = sorted(sum(correct[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return accs[int(0.025 * n_boot)], accs[int(0.975 * n_boot) - 1]


def mcnemar_exact(a, b):
    """Two-sided exact McNemar p-value on paired correct/incorrect lists."""
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    if n == 0:
        return 1.0, b01, b10
    k = min(b01, b10)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p), b01, b10


def main():
    sts = stemmers()
    print("Stemmers:", ", ".join(sts))
    for fname, title in SETS:
        gold = load_gold(fname)
        res = {}
        for name, fn in sts.items():
            t0 = time.perf_counter()
            res[name] = [fn(w) == root for w, root, _ in gold]
            res[name + "_ms"] = (time.perf_counter() - t0) * 1000
        print(f"\n## {title} ({len(gold)} words)\n")
        print("| stemmer | accuracy | 95% CI | errors | vs rapikan (McNemar p) | time |")
        print("|---|---|---|---|---|---|")
        for name in sts:
            c = res[name]
            lo, hi = boot_ci(c)
            if name == "rapikan":
                test = "-"
            else:
                p, only_r, only_o = mcnemar_exact(res["rapikan"], c)
                test = f"p={p:.3f} (rapikan-only right {only_r}, {name}-only right {only_o})"
            print(f"| {name} | {100 * sum(c) / len(c):.1f}% | {100 * lo:.1f}-{100 * hi:.1f}% "
                  f"| {len(c) - sum(c)} | {test} | {res[name + '_ms']:.0f} ms |")


if __name__ == "__main__":
    main()
