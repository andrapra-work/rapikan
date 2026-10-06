"""Matching a spoken (speech-to-text) answer against the expected answer.

Built for quiz-style games: a child says "eee... kucingnyaaa! bukan anjing",
the STT engine writes that down, and the game needs to know whether the
answer "kucing" was given.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import List, Optional

from .pipeline import Pipeline
from .presets import stt_answer


@dataclass
class AnswerMatch:
    correct: bool
    heard: List[str]
    expected: List[str]
    rejected: List[str] = field(default_factory=list)
    others: List[str] = field(default_factory=list)
    fuzzy: List[str] = field(default_factory=list)

    @property
    def ambiguous(self) -> bool:
        """True when the answer also contained other content words (a guess list, or a self-correction)."""
        return self.correct and bool(self.others)

    def reason(self) -> str:
        if not self.correct:
            if any(e in self.rejected for e in self.expected):
                return f"expected word was negated ({', '.join(self.rejected)})"
            return f"expected {self.expected}, heard {self.heard}"
        msg = "match"
        if self.fuzzy:
            msg += f" (approximate: {', '.join(self.fuzzy)})"
        if self.others:
            msg += f"; also said: {', '.join(self.others)}"
        return msg


def match_answer(spoken: str, expected: str, pipeline: Optional[Pipeline] = None,
                 fuzzy: float = 0.8) -> AnswerMatch:
    """Return whether ``spoken`` contains ``expected``.

    - Both sides go through the same normalisation (fillers, elongation, slang, numbers as words, stemming),
      so "kucingnyaaa" matches "kucing" and "2" matches "dua".
    - Words right after a negation ("bukan anjing") count as rejected, not as answers.
    - ``fuzzy`` is the minimum similarity (0-1) to accept a near miss from the STT engine; use 1.0 to disable.

    >>> match_answer("eee... kucingnyaaa! bukan anjing", "kucing").correct
    True
    >>> match_answer("bukan kucing", "kucing").correct
    False
    """
    p = pipeline or _default()
    heard_tokens = p(spoken).tokens
    want = [w.lower() for w in p(expected).words if w.lower() not in p.lexicon.negations]
    neg = p.lexicon.negations

    affirmed, rejected = [], []
    i = 0
    while i < len(heard_tokens):
        w = heard_tokens[i].text.lower()
        if w in neg and i + 1 < len(heard_tokens):
            rejected.append(heard_tokens[i + 1].text.lower())
            i += 2
            continue
        if w not in neg:
            affirmed.append(w)
        i += 1

    approx, ok = [], bool(want)
    for e in want:
        if e in affirmed:
            continue
        if fuzzy < 1.0:
            close = difflib.get_close_matches(e, affirmed, n=1, cutoff=fuzzy)
            if close and len(e) >= 4:
                approx.append(f"{close[0]}~{e}")
                continue
        ok = False
    matched = set(want) | {a.split("~")[0] for a in approx}
    others = [w for w in dict.fromkeys(affirmed) if w not in matched and w not in p.lexicon.stopwords]
    return AnswerMatch(ok, [t.text for t in heard_tokens], want, rejected, others, approx)


_P: Optional[Pipeline] = None


def _default() -> Pipeline:
    global _P
    if _P is None:
        _P = stt_answer()
    return _P
