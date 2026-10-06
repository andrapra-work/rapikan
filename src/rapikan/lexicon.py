"""Word lists used by the pipeline, with a way to plug in your own.

Bundled data:
- roots.tsv: Indonesian root words from hunspell-id (LGPL-3.0, see
  data/LICENSE-roots-LGPL-3.0.txt), with the affixes hunspell allows on each
  root (built by tools/build_roots.py). Commercial use is allowed.
- slang.tsv, stopwords.txt: written for rapikan (MIT).

Not bundled, but loadable (check their licenses for your use case):
- Kateglo root list shipped inside Sastrawi (CC-BY-NC-SA 3.0, non-commercial):
  ``Lexicon.default().with_sastrawi_roots()``
- Colloquial Indonesian Lexicon (Salsabila et al., 2018), CSV with
  slang/formal columns: ``lex.with_slang_csv(path)``
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field, replace
from functools import lru_cache
from importlib import resources
from typing import Dict, FrozenSet, Iterable, Optional

NEGATIONS: FrozenSet[str] = frozenset({
    "tidak", "bukan", "belum", "jangan", "tanpa", "tak",
    # raw informal forms, so negation is recognised even when slang
    # normalisation is switched off
    "nggak", "ngga", "nggk", "ngak", "gak", "ga", "gk", "enggak", "engga",
    "kagak", "kaga", "ndak", "nda", "tdk", "bkn", "blm", "jgn", "gpp",
})

FILLERS: FrozenSet[str] = frozenset({
    "e", "ee", "eh", "ehm", "em", "emm", "hm", "hmm", "mm", "uh", "um", "umm",
    "anu", "eem", "aa", "ah",
})

INTENSIFIERS: FrozenSet[str] = frozenset({"sangat", "sekali", "terlalu", "paling", "amat", "banget", "bgt"})


def _read_lines(name: str) -> list:
    text = resources.files("rapikan.data").joinpath(name).read_text(encoding="utf-8")
    return [l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#")]


@lru_cache(maxsize=1)
def _bundled():
    affixes: Dict[str, FrozenSet[str]] = {}
    for line in _read_lines("roots.tsv"):
        w, _, fams = line.partition("\t")
        affixes[w] = frozenset(f for f in fams.split(",") if f)
    roots = frozenset(affixes)
    slang: Dict[str, str] = {}
    for line in _read_lines("slang.tsv"):
        k, _, v = line.partition("\t")
        if k and v:
            slang[k.strip().lower()] = v.strip().lower()
    stop = frozenset(w.lower() for w in _read_lines("stopwords.txt"))
    return roots, slang, stop, affixes


@dataclass(frozen=True)
class Lexicon:
    roots: FrozenSet[str]
    slang: Dict[str, str]
    stopwords: FrozenSet[str]
    affixes: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    negations: FrozenSet[str] = NEGATIONS
    fillers: FrozenSet[str] = FILLERS
    protected: FrozenSet[str] = field(default_factory=frozenset)

    @classmethod
    def default(cls) -> "Lexicon":
        roots, slang, stop, affixes = _bundled()
        return cls(roots=roots, slang=dict(slang), stopwords=stop, affixes=affixes)

    # --- extension points -------------------------------------------------
    def with_roots(self, words: Iterable[str]) -> "Lexicon":
        return replace(self, roots=self.roots | {w.strip().lower() for w in words if w.strip()})

    def with_slang(self, mapping: Dict[str, str]) -> "Lexicon":
        merged = dict(self.slang)
        merged.update({k.lower(): v.lower() for k, v in mapping.items()})
        return replace(self, slang=merged)

    def with_stopwords(self, add: Iterable[str] = (), remove: Iterable[str] = ()) -> "Lexicon":
        return replace(self, stopwords=(self.stopwords | {w.lower() for w in add}) - {w.lower() for w in remove})

    def with_protected(self, words: Iterable[str]) -> "Lexicon":
        """Words that are never changed by slang mapping or stemming (brand names, jargon)."""
        return replace(self, protected=self.protected | {w.lower() for w in words})

    def with_slang_csv(self, path: str, informal_col: str = "slang", formal_col: str = "formal") -> "Lexicon":
        with open(path, encoding="utf-8") as f:
            rows = {r[informal_col].strip().lower(): r[formal_col].strip().lower()
                    for r in csv.DictReader(f) if r.get(informal_col) and r.get(formal_col)}
        # entries we wrote by hand win over bulk lists
        rows.update(self.slang)
        return replace(self, slang=rows)

    def with_sastrawi_roots(self) -> "Lexicon":
        """Add the Kateglo root list bundled with Sastrawi (CC-BY-NC-SA 3.0: non-commercial)."""
        try:
            data = resources.files("Sastrawi.Stemmer.data").joinpath("kata-dasar.txt").read_text(encoding="utf-8")
        except (ModuleNotFoundError, FileNotFoundError) as e:  # pragma: no cover
            raise RuntimeError("pip install Sastrawi to use its root list") from e
        return self.with_roots(data.split())

    # --- lookups ------------------------------------------------------------
    def is_known(self, word: str) -> bool:
        w = word.lower()
        return w in self.roots or w in self.slang or w in self.stopwords or w in self.negations


@lru_cache(maxsize=1)
def default_lexicon() -> Lexicon:
    return Lexicon.default()


def resolve(lexicon: Optional[Lexicon]) -> Lexicon:
    return lexicon if lexicon is not None else default_lexicon()
