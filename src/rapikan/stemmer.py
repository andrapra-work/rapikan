"""Dictionary-checked Indonesian stemmer.

Based on the affix model behind Nazief-Adriani and Enhanced Confix Stripping
(Arifin et al., 2009): words are built as

    [prefix1 [prefix2 [prefix3]]] root [derivational] [possessive] [particle]

with meN-/peN- sound changes (menunggu = meN + tunggu). Instead of applying
the ECS rules in one fixed order, this stemmer enumerates every decomposition
the affix grammar allows, keeps the ones whose root is in the dictionary, and
picks the most plausible one:

1. decompositions the dictionary licenses: the bundled hunspell-id data says
   which affixes each root takes, so 'mengirim' -> kirim (takes meN-) wins over
   irim (takes nothing), and 'pembelian' -> beli wins over belian
2. then the fewest affixes removed (kebersihan -> bersih, not sih)
3. then the order alternatives are generated in, which encodes the usual
   preferences: memakan -> makan before pakan, terasa -> rasa before asa

Words that are already in the dictionary are returned as they are. That keeps
'mereka', 'kalian', 'sebagai' intact, at the cost of a few lexicalised entries
('mengalami', 'pelari') staying unstemmed. Unknown words are left alone, except that clear clitics (-nya, -ku, -mu,
-lah, -kah) can be stripped.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

from .lexicon import Lexicon, resolve

_V = set("aeiou")

PARTICLES = ("lah", "kah", "tah", "pun")
POSSESSIVES = ("nya", "ku", "mu")
DERIVATIONAL = ("an", "kan", "i")  # -an first: masakan -> masak, not masa

# prefix pairs that never occur around the same root (ECS invalid confixes)
_INVALID_CONFIX = {("be", "i"), ("di", "an"), ("ke", "i"), ("ke", "kan"), ("me", "an"),
                   ("se", "i"), ("se", "kan"), ("te", "an")}
_OUTER_ONLY = {"di", "ke", "se", "ku", "kau"}
# stemmer prefix label -> affix family in the bundled root data (None: always allowed)
_LICENSE_FAMILY = {"me": "me", "pe": "pe", "pel": "pe", "per": "per", "ber": "be", "be": "be", "bel": "be",
                   "ter": "te", "te": "te", "di": "di", "ke": "ke", "se": "se", "ku": None, "kau": None}
MAX_PREFIXES = 3
MIN_ROOT = 2


class Stemmer:
    def __init__(self, lexicon: Optional[Lexicon] = None, strip_unknown_clitics: bool = True):
        self.lex = resolve(lexicon)
        self.strip_unknown_clitics = strip_unknown_clitics
        self._cache: Dict[str, str] = {}

    # ------------------------------------------------------------------ API
    def stem(self, word: str) -> str:
        w = word.lower()
        if w in self._cache:
            return self._cache[w]
        res = self._stem(w)
        if len(self._cache) < 200_000:
            self._cache[w] = res
        return res

    __call__ = stem

    def explain(self, word: str) -> List[Tuple[str, Tuple[str, ...], Tuple[str, ...], str]]:
        """All dictionary-valid decompositions, best first: (root, prefixes, suffixes, licensing)."""
        w = word.lower()
        cands = list(self._candidates(w))
        cands.sort(key=self._rank)
        return [(c[0], c[1], c[2], "licensed" if c[4] == 0 else f"{c[4]} unlicensed") for c in cands]

    # ------------------------------------------------------------- internals
    def _stem(self, w: str) -> str:
        if not w.isalpha() and "-" not in w:
            return w
        if w in self.lex.protected:
            return w
        if "-" in w:
            return self._stem_compound(w)
        if len(w) <= 3:
            return w
        if w in self.lex.roots:
            return w  # the dictionary is the authority (mereka, kalian, sebagai stay intact)
        cands = list(self._candidates(w))
        if cands:
            return min(cands, key=self._rank)[0]
        if self.strip_unknown_clitics:
            for suf in PARTICLES + POSSESSIVES:
                if w.endswith(suf) and len(w) - len(suf) >= 3:
                    return w[: -len(suf)]
        return w

    def _stem_compound(self, w: str) -> str:
        left, _, right = w.partition("-")
        if w in self.lex.roots and not (left == right and left in self.lex.roots):
            return w  # lexicalised compound: kupu-kupu, sayur-mayur
        if not left or not right:
            return w
        # reduplication with affixes on the second half: anak-anaknya, bermain-main
        sl, sr = self.stem(left), self.stem(right)
        if sl == sr or right.startswith(left) or sl == right or sr == left:
            return sl if sl in self.lex.roots else sr
        # 'ke-2', 'payment-api', 'sayur-mayur': leave the compound intact
        return w

    @staticmethod
    def _rank(c):
        root, prefixes, suffixes, order, unlicensed = c
        return (unlicensed, len(prefixes) + len(suffixes), order)

    def _unlicensed(self, root: str, prefixes, suffixes) -> int:
        allowed = self.lex.affixes.get(root)
        if not allowed:
            return len([p for p in prefixes if _LICENSE_FAMILY.get(p)]) + \
                len([s for s in suffixes if s in DERIVATIONAL]) or 0
        n = sum(1 for p in prefixes if _LICENSE_FAMILY.get(p) and _LICENSE_FAMILY[p] not in allowed)
        n += sum(1 for s in suffixes if s in DERIVATIONAL and s not in allowed)
        return n

    def _candidates(self, w: str) -> Iterator[tuple]:
        order = 0
        seen = set()
        for base, suffixes in self._suffix_options(w):
            for root, prefixes in self._prefix_options(base, (), 0):
                if len(root) < MIN_ROOT or root not in self.lex.roots:
                    continue
                if not prefixes and not suffixes:
                    continue
                if not self._valid(prefixes, suffixes):
                    continue
                key = (root, prefixes, suffixes)
                if key in seen:
                    continue
                seen.add(key)
                order += 1
                yield root, prefixes, suffixes, order, self._unlicensed(root, prefixes, suffixes)

    @staticmethod
    def _valid(prefixes: Tuple[str, ...], suffixes: Tuple[str, ...]) -> bool:
        deriv = next((s for s in suffixes if s in DERIVATIONAL), None)
        if prefixes and deriv:
            fam = _family(prefixes[0])
            if (fam, deriv) in _INVALID_CONFIX:
                return False
        return True

    @staticmethod
    def _suffix_options(w: str) -> List[Tuple[str, Tuple[str, ...]]]:
        opts = [(w, ())]
        for group in (PARTICLES, POSSESSIVES, DERIVATIONAL):
            new = []
            for base, sufs in opts:
                for s in group:
                    if base.endswith(s) and len(base) - len(s) >= MIN_ROOT + 1:
                        new.append((base[: -len(s)], (s,) + sufs))
            opts += new
        return opts

    def _prefix_options(self, w: str, prefixes: Tuple[str, ...], depth: int) -> Iterator[Tuple[str, Tuple[str, ...]]]:
        yield w, prefixes
        if depth >= MAX_PREFIXES:
            return
        for label, rest in _strip_prefix(w):
            if label in prefixes:
                continue
            if depth > 0 and label in _OUTER_ONLY:
                continue
            if len(rest) < MIN_ROOT:
                continue
            yield from self._prefix_options(rest, prefixes + (label,), depth + 1)


def _family(prefix: str) -> str:
    if prefix in ("ber", "be", "bel"):
        return "be"
    if prefix.startswith("me"):
        return "me"
    if prefix in ("ter", "te"):
        return "te"
    if prefix.startswith("pe"):
        return "pe"
    return prefix


def _strip_prefix(w: str) -> List[Tuple[str, str]]:
    """Every way one prefix can be removed from w, in preference order."""
    out: List[Tuple[str, str]] = []
    add = lambda label, rest: out.append((label, rest)) if rest else None

    for p in ("di", "ke", "se", "ku"):
        if w.startswith(p):
            add(p, w[2:])
    if w.startswith("kau"):
        add("kau", w[3:])

    # te-r... is tried before ter-: terasa -> rasa (not asa), terawat -> rawat
    if w.startswith("ter"):
        if len(w) > 3 and w[3] in _V:
            add("te", w[2:])
        add("ter", w[3:])
    elif w.startswith("te") and len(w) > 5 and w[2] not in _V and w[3:5] == "er":
        add("te", w[2:])  # te-C1erC2

    # ber- / be- / bel-
    if w.startswith("belajar"):
        add("bel", w[3:])
    if w.startswith("ber"):
        if len(w) > 3 and w[3] in _V:
            add("be", w[2:])  # be-renang, be-rupa
        add("ber", w[3:])
    elif w.startswith("be") and len(w) > 5 and w[2] not in _V and w[3:5] == "er":
        add("be", w[2:])  # be-kerja

    # meN-
    if w.startswith("me"):
        out.extend(("me", r) for r in _nasal_restore(w, "me"))
    # peN- / per- / pel-
    if w.startswith("pe"):
        if w.startswith("pelajar"):
            add("pel", w[3:])
        if w.startswith("per"):
            if len(w) > 3 and w[3] in _V:
                add("pe", w[2:])  # pe-rokok, pe-rawat
            add("per", w[3:])
        elif len(w) > 5 and w[2] not in _V and w[3:5] == "er":
            add("pe", w[2:])  # pe-kerja
        out.extend(("pe", r) for r in _nasal_restore(w, "pe"))
    return out


def _nasal_restore(w: str, head: str) -> List[str]:
    """Undo meN-/peN- assimilation. head is 'me' or 'pe'."""
    r: List[str] = []
    n = len(head)
    if w.startswith(head + "nge") and len(w) > n + 4:
        r.append(w[n + 3:])  # menge-cat
    if w.startswith(head + "ng"):
        rest = w[n + 2:]
        if rest:
            if rest[0] in _V:
                r += [rest, "k" + rest]  # meng-ambil, meng-(k)irim
            elif rest[0] in "ghkq":
                r.append(rest)
    elif w.startswith(head + "ny"):
        rest = w[n + 2:]
        if rest and rest[0] in _V:
            r += ["s" + rest, "ny" + rest]  # meny-(s)apu, me-nyanyi
    elif w.startswith(head + "m"):
        rest = w[n + 1:]
        if rest:
            if rest[0] in "bfv" or rest.startswith(("pr", "pl")):
                r.append(rest)  # mem-baca, mem-produksi
            elif rest[0] == "p":
                r.append(rest)
            elif rest[0] in _V:
                r += ["m" + rest, "p" + rest]  # me-makan, mem-(p)ukul
    elif w.startswith(head + "n"):
        rest = w[n + 1:]
        if rest:
            if rest[0] in "cdjz" or rest.startswith("sy"):
                r.append(rest)  # men-cari, men-syukuri
            elif rest[0] == "t" and len(rest) > 1 and rest[1] not in _V:
                r.append(rest)  # men-transfer
            elif rest[0] in _V:
                r += ["n" + rest, "t" + rest]  # me-nikah, men-(t)unggu
    rest = w[n:]
    if rest and rest[0] in "lmnrwy":
        r.append(rest)  # me-lihat, me-rasa, pe-lari
    return r


_default: Optional[Stemmer] = None


def stem(word: str) -> str:
    """Stem one word with the default dictionary."""
    global _default
    if _default is None:
        _default = Stemmer()
    return _default.stem(word)
