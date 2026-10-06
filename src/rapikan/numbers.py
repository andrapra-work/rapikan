"""Numbers to Indonesian words, for text-to-speech and spoken-answer matching."""
from __future__ import annotations

import re
from typing import Optional

_ONES = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan",
         "sepuluh", "sebelas"]
_SCALES = [(10 ** 12, "triliun"), (10 ** 9, "miliar"), (10 ** 6, "juta"), (1000, "ribu")]


def terbilang(n: int) -> str:
    """Spell out an integer in Indonesian.

    >>> terbilang(28000)
    'dua puluh delapan ribu'
    >>> terbilang(1105)
    'seribu seratus lima'
    """
    if n < 0:
        return "minus " + terbilang(-n)
    if n < 12:
        return _ONES[n]
    if n < 20:
        return _ONES[n - 10] + " belas"
    if n < 100:
        tens, rest = divmod(n, 10)
        return _ONES[tens] + " puluh" + (" " + _ONES[rest] if rest else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = "seratus" if hundreds == 1 else _ONES[hundreds] + " ratus"
        return head + (" " + terbilang(rest) if rest else "")
    for size, name in _SCALES:
        if n >= size:
            big, rest = divmod(n, size)
            head = "seribu" if (size == 1000 and big == 1) else terbilang(big) + " " + name
            return head + (" " + terbilang(rest) if rest else "")
    raise ValueError(n)  # pragma: no cover


def _digits(s: str) -> str:
    return " ".join(_ONES[int(c)] for c in s)


def _parse_id_number(s: str) -> Optional[float]:
    """Parse '70.000', '2,5', '1.250.000' using Indonesian separators."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
        return float(s.replace(".", ""))
    if re.fullmatch(r"\d+,\d+", s):
        return float(s.replace(",", "."))
    if re.fullmatch(r"\d+\.\d{1,2}", s):  # 2.5 written English-style
        return float(s)
    if re.fullmatch(r"\d+", s):
        return float(s)
    return None


def number_words(n: float) -> str:
    if float(n).is_integer():
        return terbilang(int(n))
    whole, frac = f"{n}".split(".")
    return terbilang(int(whole)) + " koma " + _digits(frac)


_MULT = {"rb": 1000, "ribu": 1000, "k": 1000, "jt": 10 ** 6, "juta": 10 ** 6,
         "m": 10 ** 9, "miliar": 10 ** 9, "t": 10 ** 12, "triliun": 10 ** 12}
_CURRENCY = {"rp": "rupiah", "idr": "rupiah", "usd": "dolar", "$": "dolar"}


def spell_token(text: str) -> Optional[str]:
    """Spell a number or money token, or return None if it isn't one we can read."""
    raw = text.strip()
    low = raw.lower()

    m = re.fullmatch(r"(rp\.?|idr|usd|\$)?\s?(\d[\d.,]*)\s?(rb|ribu|k|jt|juta|m|miliar|t|triliun)?", low)
    if m and (m.group(1) or m.group(3)):
        cur = _CURRENCY.get((m.group(1) or "").rstrip("."), "")
        val = _parse_id_number(m.group(2))
        if val is None:
            return None
        val *= _MULT.get(m.group(3) or "", 1)
        return (number_words(val) + (" " + cur if cur else "")).strip()

    if re.fullmatch(r"\d+%", low):
        return number_words(float(low[:-1])) + " persen"
    m = re.fullmatch(r"(\d+)x", low)
    if m:
        return terbilang(int(m.group(1))) + " kali"
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", low)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        return terbilang(h) + (" " + terbilang(mi) if mi else "")
    m = re.fullmatch(r"(\d+)/(\d+)", low)
    if m:
        return terbilang(int(m.group(1))) + " per " + terbilang(int(m.group(2)))
    m = re.fullmatch(r"(\d+)-(\d+)", low)
    if m:  # scores and ranges: 3-0, 2-1, 10-12
        return terbilang(int(m.group(1))) + " " + terbilang(int(m.group(2)))
    val = _parse_id_number(low)
    if val is not None:
        if re.fullmatch(r"0\d+", low):  # phone numbers, codes: read digit by digit
            return _digits(low)
        return number_words(val)
    return None


def ordinal(word: str) -> Optional[str]:
    """'ke-2' -> 'kedua', 'ke-1' -> 'pertama'."""
    m = re.fullmatch(r"ke-?(\d+)", word.lower())
    if not m:
        return None
    n = int(m.group(1))
    return "pertama" if n == 1 else "ke" + terbilang(n)
