"""Build an independent stemming gold set from Indonesian UD treebanks.

UD_Indonesian-CSUI (CC BY-SA 4.0) annotates lemmas as roots
(mempertimbangkan -> timbang, bagian -> bagi), which is what a stemmer outputs.

Rules:
- Use the surface word as written, including clitic multiword tokens
  (menyetujuinya -> setuju, the lemma of its first part).
- Keep NOUN, VERB, ADJ, ADV; alphabetic, lowercase, 3+ letters.
- Drop words annotated with more than one lemma (ambiguous in the treebank).
- Keep only rows where lemma != surface. Rows with lemma == surface are
  dropped: the treebank often leaves affixed words unlemmatised.

    python tools/ud_to_gold.py path/to/id_csui-ud-train.conllu ... > bench/ud_csui_stems.tsv
"""
from __future__ import annotations

import sys
from collections import defaultdict

KEEP = {"NOUN", "VERB", "ADJ", "ADV"}


def read(paths):
    pairs = defaultdict(lambda: defaultdict(int))
    for path in paths:
        mwt_end, mwt_form, first = 0, None, False
        for line in open(path, encoding="utf-8"):
            if not line.strip() or line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            tid = c[0]
            if "-" in tid:
                a, b = map(int, tid.split("-"))
                mwt_end, mwt_form, first = b, c[1], True
                continue
            if "." in tid:
                continue
            i = int(tid)
            form, lemma, upos = c[1], c[2], c[3]
            if mwt_form is not None and i <= mwt_end:
                if first:
                    form, first = mwt_form, False
                else:
                    continue
                if i == mwt_end:
                    mwt_form = None
            if upos not in KEEP:
                continue
            w, l = form.lower(), lemma.lower()
            if not (w.isalpha() and l.isalpha() and len(w) >= 3) or form != w:
                continue
            pairs[w][l] += 1
    return pairs


def main():
    pairs = read(sys.argv[1:])
    print("# Built by tools/ud_to_gold.py from UD_Indonesian-CSUI (CC BY-SA 4.0).")
    print("# Independent of rapikan: lemmas annotated by the treebank authors.")
    print("# word\texpected_root\tcategory")
    kept = amb = 0
    for w in sorted(pairs):
        lemmas = pairs[w]
        if len(lemmas) > 1:
            amb += 1
            continue
        (l,) = lemmas
        if w == l:
            # Unreliable: many clearly affixed words (keterlambatan, diperoleh)
            # carry their surface form as lemma, so "unchanged" rows are not gold.
            continue
        print(f"{w}\t{l}\taffixed")
        kept += 1
    print(f"kept {kept}, dropped {amb} ambiguous", file=sys.stderr)


if __name__ == "__main__":
    main()
