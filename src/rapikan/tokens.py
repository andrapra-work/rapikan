"""Typed tokenizer for informal Indonesian text.

Each token knows what it is (word, number, money, emoji, url, ...), so later
steps can decide per type instead of guessing from the characters.
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List

URL = "url"
EMAIL = "email"
MENTION = "mention"
HASHTAG = "hashtag"
EMOJI = "emoji"
EMOTICON = "emoticon"
MONEY = "money"
NUMBER = "number"
WORD = "word"
PUNCT = "punct"

ALL_TYPES = (URL, EMAIL, MENTION, HASHTAG, EMOJI, EMOTICON, MONEY, NUMBER, WORD, PUNCT)


@dataclass
class Token:
    text: str
    type: str
    orig: str = ""
    start: int = -1
    end: int = -1
    tags: set = field(default_factory=set)

    def __post_init__(self):
        if not self.orig:
            self.orig = self.text

    def copy(self, **changes) -> "Token":
        t = Token(self.text, self.type, self.orig, self.start, self.end, set(self.tags))
        for k, v in changes.items():
            setattr(t, k, v)
        return t

    def __repr__(self) -> str:  # compact, readable in notebooks
        tags = ("," + ",".join(sorted(self.tags))) if self.tags else ""
        return f"<{self.type}{tags}:{self.text}>"


_EMO_CHAR = "[⌀-⏿☀-➿⬀-⯿\U0001F000-\U0001FAFF]"
_EMOJI = (
    r"(?:[\U0001F1E6-\U0001F1FF]{2})"  # flags
    rf"|(?:{_EMO_CHAR}️?[\U0001F3FB-\U0001F3FF]?(?:‍{_EMO_CHAR}️?[\U0001F3FB-\U0001F3FF]?)*)"
)

_PATTERNS = [
    (URL, r"(?:https?://|www\.)[^\s<>\"']+"),
    (EMAIL, r"[\w.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]+"),
    (MENTION, r"(?<![\w@])@[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?"),
    (HASHTAG, r"(?<![\w#])#\w+"),
    (EMOJI, _EMOJI),
    (EMOTICON, r"(?<!\w)(?:[:;=]['\-^]?[)(DPp/\\|*3]+|<3|\^_\^|T_T)(?!\w)"),
    (MONEY, r"(?:(?:Rp\.?|IDR|USD|\$)\s?\d+(?:[.,]\d+)*(?:\s?(?:rb|ribu|k|jt|juta|m|miliar|t|triliun)\b)?"
            r"|\d+(?:[.,]\d+)?\s?(?:rb|ribu|jt|juta|k)\b)"),
    (NUMBER, r"\d+(?:[.,:/\-]\d+)*(?:%|[A-Za-z]+)?"),
    (WORD, r"[^\W\d_][\w]*(?:[-'’][\w]+)*"),
    (PUNCT, r"\.{2,}|…|[!?]{2,}|[^\w\s]"),
]
_MASTER = re.compile("|".join(f"(?P<{name}_{i}>{p})" for i, (name, p) in enumerate(_PATTERNS)), re.UNICODE)
_URL_TRAIL = re.compile(r"[.,!?;:)\]]+$")

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌⁠﻿"), None)


def clean_text(text: str) -> str:
    """Unicode and markup cleanup that never removes meaning.

    Unescapes HTML entities, applies NFKC (fullwidth chars, ligatures), turns
    smart quotes into plain ones, drops zero-width characters (but keeps the
    zero-width joiner that glues emoji together), and collapses whitespace.
    """
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ZERO_WIDTH)
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return text.strip()


def tokenize(text: str, clean: bool = True) -> List[Token]:
    """Split text into typed tokens.

    >>> [t.text for t in tokenize("Harga naik jd Rp28.000, skor 3-0!!!")]
    ['Harga', 'naik', 'jd', 'Rp28.000', ',', 'skor', '3-0', '!!!']
    """
    if clean:
        text = clean_text(text)
    out: List[Token] = []
    for m in _MASTER.finditer(text):
        kind = m.lastgroup.rsplit("_", 1)[0]
        s, e = m.span()
        t = m.group()
        if kind == URL:
            trail = _URL_TRAIL.search(t)
            if trail:
                t = t[: trail.start()]
                e = s + len(t)
                out.append(Token(t, URL, start=s, end=e))
                for i, c in enumerate(trail.group()):
                    out.append(Token(c, PUNCT, start=e + i, end=e + i + 1))
                continue
        tok = Token(t, kind, start=s, end=e)
        if kind == WORD:
            _tag_word(tok)
        out.append(tok)
    return out


_CAMEL = re.compile(r"[a-z][A-Z]|[A-Z]{2,}[a-z]")


def _tag_word(tok: Token) -> None:
    t = tok.text
    letters = [c for c in t if c.isalpha()]
    if len(letters) >= 2 and all(c.isupper() for c in letters):
        tok.tags.add("caps")
    if _CAMEL.search(t) or (any(c.isdigit() for c in t) and not re.fullmatch(r"[^\W\d_]+2[^\W\d_]*", t)):
        tok.tags.add("technical")
    if "-" in t and not _is_reduplication(t):
        tok.tags.add("compound")


def _is_reduplication(word: str) -> bool:
    parts = word.lower().split("-")
    return len(parts) == 2 and (parts[1].startswith(parts[0]) or parts[0] == parts[1])


def detokenize(tokens: List[Token]) -> str:
    """Join tokens back into readable text (no space before . , ! ? etc.)."""
    out = ""
    quote_open = False
    glue_next = False
    for i, t in enumerate(tokens):
        is_quote = t.type == PUNCT and t.text in ('"', "'")
        closing = is_quote and quote_open
        attach = (t.type == PUNCT and t.text[0] in ".,!?;:)]…%") or closing or glue_next
        if i and not attach:
            out += " "
        out += t.text
        glue_next = t.text in ("(", "[") or (is_quote and not quote_open)
        if is_quote:
            quote_open = not quote_open
    return out
