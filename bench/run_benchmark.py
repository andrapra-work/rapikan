"""Benchmark rapikan against Sastrawi.

Two parts:
1. Stemming accuracy on bench/gold_stems.tsv (hand-written, KBBI roots).
2. Meaning preservation on sentences: does the pipeline keep negation,
   numbers and emoji, and keep special tokens in one piece?

The Sastrawi side is the pipeline most tutorials use: lowercase, regex away
everything that is not a letter or digit, Sastrawi stopword remover,
Sastrawi stemmer.

    pip install -e .[bench]
    python bench/run_benchmark.py
"""
from __future__ import annotations

import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from rapikan import Stemmer, presets, tokenize

HERE = Path(__file__).parent


def load_gold(name="gold_stems.tsv"):
    rows = []
    for line in (HERE / name).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        w, root, cat = line.split("\t")
        rows.append((w, root, cat))
    return rows


def stem_benchmark(show_errors=True, gold_file="gold_stems.tsv", title="Stemming accuracy"):
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    sas = StemmerFactory().create_stemmer()
    ours = Stemmer()
    gold = load_gold(gold_file)

    res = {"rapikan": defaultdict(lambda: [0, 0]), "sastrawi": defaultdict(lambda: [0, 0])}
    errors = {"rapikan": [], "sastrawi": []}
    timing = {}
    for name, fn in (("rapikan", ours.stem), ("sastrawi", sas.stem)):
        t0 = time.perf_counter()
        for w, root, cat in gold:
            got = fn(w)
            ok = got == root
            res[name][cat][1] += 1
            res[name]["ALL"][1] += 1
            if ok:
                res[name][cat][0] += 1
                res[name]["ALL"][0] += 1
            else:
                errors[name].append(f"{w} -> {got} (want {root})")
        timing[name] = (time.perf_counter() - t0) * 1000

    cats = sorted({c for _, _, c in gold}, key=lambda c: [g[2] for g in gold].index(c)) + ["ALL"]
    print(f"\n## {title} ({len(gold)} words)\n")
    print(f"| {'category':<15} | {'n':>3} | {'rapikan':>8} | {'Sastrawi':>8} |")
    print(f"|{'-' * 17}|{'-' * 5}|{'-' * 10}|{'-' * 10}|")
    for c in cats:
        a, n = res["rapikan"][c]
        b, _ = res["sastrawi"][c]
        name = f"**{c}**" if c == "ALL" else c
        print(f"| {name:<15} | {n:>3} | {100 * a / n:>7.1f}% | {100 * b / n:>7.1f}% |")
    print(f"\nTime for {len(gold)} words (cold cache): rapikan {timing['rapikan']:.0f} ms, Sastrawi {timing['sastrawi']:.0f} ms")
    if show_errors:
        for name in ("rapikan", "sastrawi"):
            print(f"\n{name} errors ({len(errors[name])}): " + "; ".join(errors[name]))
    return res


# ------------------------------------------------------------ meaning tests

CASES = [
    # (text, expected substrings in the output that carry the meaning)
    ("Kuahnya nggak enak 😞", {"negation": "tidak_enak", "emoji": "😞"}),
    ("Pelayanannya gak ramah, tp bakminya enak bgt 😍", {"negation": "tidak_ramah", "emoji": "😍"}),
    ("Pemainnya nggak kenal lelah!!!", {"negation": "tidak_kenal"}),
    ("Skornya 8/10, nggak jelek kan?", {"negation": "tidak_jelek"}),
    ("Rasanya bukan yang terbaik, belum layak harga segitu", {"negation": "bukan_terbaik"}),
    ("Jangan pesan yang pedas 🌶️", {"negation": "jangan_pesan", "emoji": "🌶️"}),
    ("Ga suka sama sekali", {"negation": "tidak_suka"}),
    ("Tdk recommended, kecewa 😤", {"negation": "tidak_recommended", "emoji": "😤"}),
]

NEGATION_FORMS = {"tidak", "bukan", "belum", "jangan", "nggak", "gak", "ga", "tdk", "tak"}

TOKEN_CASES = [
    ("Harga naik jd Rp28.000", "Rp28.000"),
    ("Menang 3-0 lawan Vietnam", "3-0"),
    ("Skornya 8/10", "8/10"),
    ("Pod payment-api restart 15x", "payment-api"),
    ("Pod payment-api restart 15x", "15x"),
    ("Timnas U-23 menang", "U-23"),
    ("Penonton 70.000 orang", "70.000"),
    ("Mulai jam 19:30", "19:30"),
    ("Diskon 10% hari ini", "10%"),
    ("Cek https://grafana.internal/d/pay sekarang", "https://grafana.internal/d/pay"),
]


def naive_sastrawi():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    stemmer = StemmerFactory().create_stemmer()
    stop = set(StopWordRemoverFactory().get_stop_words())

    def run(text):
        text = text.lower()
        text = re.sub(r"https?://\S+|@\w+", " ", text)
        words = re.findall(r"[a-z0-9]+", text)
        words = [w for w in words if w not in stop]
        return [stemmer.stem(w) for w in words]
    return run, stop


def meaning_benchmark():
    naive, sas_stop = naive_sastrawi()
    ours = presets.sentiment()
    print("\n## Meaning preservation (sentiment use case)\n")
    print("Sastrawi stopword list contains:", ", ".join(sorted(w for w in ("tidak", "bukan", "belum", "jangan") if w in sas_stop)) or "none of the negations")
    print()
    print("| text | rapikan (sentiment preset) | tutorial pipeline (Sastrawi) |")
    print("|---|---|---|")
    keep = {"rapikan": [0, 0], "naive": [0, 0]}
    for text, expect in CASES:
        r = ours(text).words
        n = naive(text)
        for key, want in expect.items():
            keep["rapikan"][1] += 1
            keep["naive"][1] += 1
            if want in r:
                keep["rapikan"][0] += 1
            if key == "negation":
                # generous to the tutorial pipeline: any surviving negation form counts
                kept = any(x in n for x in NEGATION_FORMS)
            else:
                kept = want in n
            keep["naive"][0] += kept
        print(f"| {text} | {' '.join(r)} | {' '.join(n)} |")
    for k, (a, b) in keep.items():
        print(f"\n{'rapikan' if k == 'rapikan' else 'tutorial pipeline'}: kept {a}/{b} meaning signals (negation, emoji)")

    print("\n## Special tokens kept in one piece\n")
    ok_r = ok_n = 0
    for text, tok in TOKEN_CASES:
        ours_tokens = [t.text for t in tokenize(text)]
        naive_tokens = re.findall(r"[a-z0-9]+", text.lower())
        a = tok in ours_tokens
        b = tok.lower() in naive_tokens
        ok_r += a
        ok_n += b
        print(f"- {tok:<34} rapikan: {'yes' if a else 'NO '}   regex [a-z0-9]+: {'yes' if b else 'no'}")
    print(f"\nrapikan {ok_r}/{len(TOKEN_CASES)}, regex tokenizer {ok_n}/{len(TOKEN_CASES)}")


if __name__ == "__main__":
    quiet = "--quiet" in sys.argv
    stem_benchmark(not quiet, "gold_stems.tsv", "Stemming accuracy, development set (used while tuning)")
    stem_benchmark(not quiet, "heldout_stems.tsv", "Stemming accuracy, held-out set (never tuned against)")
    meaning_benchmark()
