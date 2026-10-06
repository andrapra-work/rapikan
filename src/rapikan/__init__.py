"""rapikan: explainable preprocessing for informal Indonesian text.

    >>> import rapikan
    >>> r = rapikan.presets.sentiment()("Kuahnya nggak enak 😞")
    >>> r.words
    ['kuahnya', 'tidak_enak', '😞']
"""
from . import presets
from .answer import AnswerMatch, match_answer
from .lexicon import Lexicon
from .numbers import terbilang
from .pipeline import (Elongation, Fillers, Hashtags, Lowercase, Negation, Numbers, Pipeline, Reduplication,
                       Remove, RemovePunct, Result, Slang, Stem, Step, Stopwords, TidyPunct, process)
from .stemmer import Stemmer, stem
from .tokens import Token, clean_text, detokenize, tokenize


def normalize_for_tts(text: str) -> str:
    """Text ready to be read aloud: 'Harga naik jd Rp28.000!!!' -> 'Harga naik jadi dua puluh delapan ribu rupiah!'"""
    return presets.tts()(text).text


__all__ = [
    "presets", "Pipeline", "Result", "Step", "Lexicon", "Stemmer", "Token",
    "tokenize", "detokenize", "clean_text", "stem", "terbilang", "process",
    "match_answer", "AnswerMatch", "normalize_for_tts",
    "Remove", "Hashtags", "RemovePunct", "TidyPunct", "Lowercase", "Elongation", "Reduplication",
    "Slang", "Fillers", "Numbers", "Negation", "Stopwords", "Stem",
]
__version__ = "0.1.0"
