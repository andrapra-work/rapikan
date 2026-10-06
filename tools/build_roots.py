"""Build src/rapikan/data/roots.tsv from the hunspell-id dictionary (LGPL-3.0).

For each root it records which affix families hunspell allows on it
(me, pe, per, be, te, di, ke, se for prefixes; an, kan, i for suffixes).
The stemmer uses this to prefer decompositions the dictionary licenses:
'mengirim' -> kirim (kirim takes meN-) rather than irim (takes nothing).

Usage:
    git clone --depth 1 https://github.com/LibreOffice/dictionaries
    python tools/build_roots.py dictionaries/id/id_ID.dic dictionaries/id/id_ID.aff src/rapikan/data/roots.tsv
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict

PREFIX_MORPHS = [
    ("menge", "me"), ("meng", "me"), ("meny", "me"), ("mem", "me"), ("men", "me"), ("me", "me"),
    ("penge", "pe"), ("peng", "pe"), ("peny", "pe"), ("pem", "pe"), ("pen", "pe"),
    ("per", "per"), ("pel", "pe"), ("pe", "pe"),
    ("ber", "be"), ("bel", "be"), ("be", "be"),
    ("ter", "te"), ("te", "te"),
    ("di", "di"), ("ke", "ke"), ("se", "se"),
]


def prefix_families(add: str) -> set:
    fams, s = set(), add.lower()
    while s:
        for morph, fam in PREFIX_MORPHS:
            if s.startswith(morph):
                fams.add(fam)
                s = s[len(morph):]
                break
        else:
            break
    return fams


def suffix_families(add: str) -> set:
    s = add.lower()
    if s.startswith("kan"):
        return {"kan"}
    if s.startswith("an"):
        return {"an"}
    if s.startswith("i") and not s.startswith("ilah") or s in ("i", "ilah", "ikah"):
        return {"i"}
    return set()


def split_flags(s: str) -> list:
    return [s[i:i + 2] for i in range(0, len(s), 2)]  # FLAG long


def parse_aff(path: str):
    own = defaultdict(set)    # flag -> families of its own add strings
    cont = defaultdict(set)   # flag -> continuation flags
    header = re.compile(r"^(PFX|SFX)\s+(\S+)\s+[YN]\s+\d+")
    with open(path, encoding="iso-8859-1") as f:
        for line in f:
            line = line.split("#", 1)[0].rstrip()
            if not line or header.match(line):
                continue
            parts = line.split()
            if len(parts) < 4 or parts[0] not in ("PFX", "SFX"):
                continue
            kind, flag, _strip, add = parts[:4]
            add, _, cflags = add.partition("/")
            add = "" if add == "0" else add
            own[flag] |= prefix_families(add) if kind == "PFX" else suffix_families(add)
            for c in split_flags(cflags):
                cont[flag].add(c)
    return own, cont


def flag_families(flag, own, cont, depth=0) -> set:
    fams = set(own.get(flag, ()))
    if depth < 2:
        for c in cont.get(flag, ()):
            fams |= flag_families(c, own, cont, depth + 1)
    return fams


def main(dic, aff, out):
    own, cont = parse_aff(aff)
    rows = {}
    with open(dic, encoding="iso-8859-1") as f:
        next(f)
        for line in f:
            word, _, flags = line.strip().partition("/")
            if not re.fullmatch(r"[a-z]+(-[a-z]+)?", word):
                continue
            fams = set()
            for fl in split_flags(flags):
                fams |= flag_families(fl, own, cont)
            rows[word] = rows.get(word, set()) | fams
    order = ["me", "pe", "per", "be", "te", "di", "ke", "se", "an", "kan", "i"]
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Derived from hunspell-id (LGPL-3.0). root<TAB>affix families hunspell allows\n")
        for w in sorted(rows):
            f.write(w + "\t" + ",".join(x for x in order if x in rows[w]) + "\n")
    flagged = sum(1 for v in rows.values() if v)
    print(f"{len(rows)} roots, {flagged} with affix info -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
