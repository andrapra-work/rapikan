"""Word-level normalisation for informal Indonesian.

Every function takes a single word and returns its normalised form (or None
when nothing applies), so steps stay easy to test and to explain.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Optional

from .lexicon import Lexicon

_VOWELS = set("aeiou")
_RUN3 = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)
_RUN2 = re.compile(r"([^\W\d_])\1", re.UNICODE)


def collapse_elongation(word: str, lex: Lexicon) -> Optional[str]:
    """'enakkk' -> 'enak', 'GOOLLL' -> 'GOL', 'kerennn' -> 'keren'.

    Runs of 3+ letters shrink to one. If the result is still unknown, double
    letters are tried too ('gool' -> 'gol'), but only when that produces a
    known word, so legitimate doubles ('saat', 'taat') survive.
    """
    once = _RUN3.sub(r"\1", word)
    if once == word:
        return None  # no real elongation: never touch legitimate doubles
    if lex.is_known(once):
        return once
    candidates = {once[: m.start()] + m.group(1) + once[m.end():] for m in _RUN2.finditer(once)}
    candidates.add(_RUN2.sub(r"\1", once))
    known = [c for c in candidates if lex.is_known(c)]
    return max(known, key=len) if known else once


_REDUP = re.compile(r"^([^\W\d_]+)2([^\W\d_]*)$", re.UNICODE)


def expand_reduplication(word: str) -> Optional[str]:
    """'jelek2' -> 'jelek-jelek', 'anak2nya' -> 'anak-anaknya'."""
    m = _REDUP.match(word)
    if not m:
        return None
    base, suffix = m.group(1), m.group(2)
    return f"{base}-{base}{suffix}"


# --- colloquial verb morphology ------------------------------------------

def men_prefix(root: str) -> str:
    """Attach the formal meN- prefix with correct sound changes.

    >>> men_prefix("tunggu"), men_prefix("cari"), men_prefix("pakai"), men_prefix("cat")
    ('menunggu', 'mencari', 'memakai', 'mengecat')
    """
    r = root.lower()
    if _syllables(r) == 1:
        return "menge" + r
    c = r[0]
    if c in _VOWELS or c in "gh":
        return "meng" + r
    if c == "k":
        return "meng" + r[1:] if len(r) > 1 and r[1] in _VOWELS else "meng" + r
    if c in "bfv":
        return "mem" + r
    if c == "p":
        return "mem" + r[1:] if len(r) > 1 and r[1] in _VOWELS else "mem" + r
    if c in "cdjz":
        return "men" + r
    if c == "t":
        return "men" + r[1:] if len(r) > 1 and r[1] in _VOWELS else "men" + r
    if c == "s":
        return "meny" + r[1:] if len(r) > 1 and r[1] in _VOWELS else "men" + r
    return "me" + r


def _syllables(w: str) -> int:
    return len(re.findall(r"[aeiou]+", w))


def colloquial_root(word: str, lex: Lexicon) -> Optional[str]:
    """Recover the root of colloquial nasal verbs: 'ngantri' -> 'antri',
    'nyari' -> 'cari', 'nunggu' -> 'tunggu', 'ngerjain' -> 'kerja'.

    Returns the root (not the formal verb) or None. Only accepts a candidate
    that is a known root, and only when the word itself is unknown.
    """
    w = word.lower()
    if lex.is_known(w) or len(w) < 4:
        return None
    bases = [w]
    for suf in ("in", "kan", "an"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            bases.append(w[: -len(suf)])
    for b in bases:
        for cand in _nasal_candidates(b):
            if cand in lex.roots and len(cand) >= 3:
                return cand
    return None


def _nasal_candidates(w: str) -> List[str]:
    out = []
    if w.startswith("nge") and len(w) > 4:
        out.append(w[3:])
    if w.startswith("ng") and len(w) > 3:
        rest = w[2:]
        out += [rest, "k" + rest] if rest[0] in _VOWELS else [rest]
    elif w.startswith("ny") and len(w) > 3 and w[2] in _VOWELS:
        out += ["c" + w[2:], "s" + w[2:]]
    elif w.startswith("n") and len(w) > 2 and w[1] in _VOWELS:
        out.append("t" + w[1:])
    elif w.startswith("m") and len(w) > 2 and w[1] in _VOWELS:
        out.append("p" + w[1:])
    return out


def formal_from_colloquial(word: str, lex: Lexicon) -> Optional[str]:
    """'ngantri' -> 'mengantri', 'nyari' -> 'mencari', 'ngerjain' -> 'mengerjakan'."""
    root = colloquial_root(word, lex)
    if not root:
        return None
    w = word.lower()
    suffix = "kan" if w.endswith(("in", "kan")) and not root.endswith("in") else ""
    if w.endswith("an") and not w.endswith("kan") and not suffix:
        return None  # 'ngomongan' and friends: leave to the dictionary
    return men_prefix(root) + suffix


# --- hashtags -------------------------------------------------------------

_CAMEL_SPLIT = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def split_hashtag(tag: str, lex: Lexicon, is_word=None) -> List[str]:
    """'#SemangatGaruda' -> ['semangat', 'garuda'], '#kulinersolo' -> ['kuliner', 'solo']."""
    body = tag.lstrip("#")
    if any(c.isupper() for c in body[1:]):
        return [p.lower() for p in _CAMEL_SPLIT.findall(body)]
    if "_" in body:
        return [p.lower() for p in body.split("_") if p]
    seg = _segment(body.lower(), is_word or lex.is_known)
    return seg if seg else [body.lower()]


def _segment(s: str, is_word) -> Optional[List[str]]:
    """Dictionary segmentation that prefers fewer, longer words."""
    vocab_ok = lambda w: len(w) >= 3 and is_word(w)

    @lru_cache(maxsize=None)
    def best(i: int):
        if i == len(s):
            return ()
        options = []
        for j in range(len(s), i, -1):
            piece = s[i:j]
            if vocab_ok(piece):
                rest = best(j)
                if rest is not None:
                    options.append((piece,) + rest)
        return min(options, key=len) if options else None

    res = best(0)
    return list(res) if res and len(res) > 1 else None
