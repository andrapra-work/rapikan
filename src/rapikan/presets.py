"""Ready-made pipelines. Each one is tuned for a task, because the right amount
of cleaning depends on what the model has to learn.

Every preset returns a fresh Pipeline; pass ``lexicon=`` to use your own word lists.
"""
from __future__ import annotations

from typing import Optional

from .lexicon import Lexicon
from .pipeline import (Elongation, Fillers, Hashtags, Lowercase, Negation, Numbers, Pipeline, Reduplication,
                       Remove, RemovePunct, Slang, Stem, Stopwords, TidyPunct)
from .tokens import EMAIL, EMOJI, EMOTICON, MENTION, URL


def sentiment(lexicon: Optional[Lexicon] = None) -> Pipeline:
    """Positive/negative classification with bag-of-words or TF-IDF models.
    Keeps emoji, merges negation ('tidak_enak'), no stopword removal, no stemming."""
    return Pipeline([
        Remove(URL, EMAIL, MENTION, name="remove_links_mentions"),
        Hashtags("split"),
        RemovePunct(),
        Lowercase(),
        Elongation(),
        Reduplication(),
        Slang(),
        Numbers("remove"),
        Negation("merge"),
    ], lexicon, name="sentiment")


def topic(lexicon: Optional[Lexicon] = None) -> Pipeline:
    """Topic or complaint-category classification, clustering, keyword counts.
    Compresses vocabulary hard: stopwords out, stemming on, numbers and hashtags kept."""
    return Pipeline([
        Remove(URL, EMAIL, MENTION, name="remove_links_mentions"),
        Hashtags("split"),
        Remove(EMOJI, EMOTICON, name="remove_emoji"),
        RemovePunct(),
        Lowercase(),
        Elongation(),
        Reduplication(),
        Slang(),
        Numbers("keep"),
        Negation("merge"),
        Stopwords(keep_negation=True),
        Stem(),
    ], lexicon, name="topic")


def search(lexicon: Optional[Lexicon] = None) -> Pipeline:
    """Indexing and querying text for keyword search. Same treatment for documents and queries."""
    return Pipeline([
        Remove(URL, EMAIL, name="remove_links"),
        Hashtags("split"),
        Remove(EMOJI, EMOTICON, MENTION, name="remove_emoji_mentions"),
        RemovePunct(),
        Lowercase(),
        Elongation(),
        Reduplication(),
        Slang(),
        Numbers("keep"),
        Stopwords(keep_negation=False),
        Stem(),
    ], lexicon, name="search")


def stt_answer(lexicon: Optional[Lexicon] = None) -> Pipeline:
    """Speech-to-text output to compare against an expected answer (e.g. a kids' quiz game).
    Drops fillers, spells numbers as words so '2' matches 'dua', keeps negation."""
    return Pipeline([
        Remove(URL, EMAIL, MENTION, EMOJI, EMOTICON, name="remove_non_speech"),
        Hashtags("split"),
        RemovePunct(),
        Lowercase(),
        Elongation(),
        Reduplication(),
        Slang(),
        Fillers(),
        Numbers("words"),
        Stem(),
    ], lexicon, name="stt_answer")


def tts(lexicon: Optional[Lexicon] = None) -> Pipeline:
    """Text to be read aloud by a TTS engine: numbers and money spelled out, slang made formal,
    emoji and links dropped, punctuation kept (tidied) for natural pauses. Use ``.text``."""
    return Pipeline([
        Remove(URL, EMAIL, EMOJI, EMOTICON, MENTION, name="remove_unspeakable"),
        Hashtags("remove"),
        TidyPunct(),
        Elongation(),
        Lowercase("known_caps"),
        Reduplication(),
        Slang(),
        Numbers("words"),
    ], lexicon, name="tts")


def llm(lexicon: Optional[Lexicon] = None) -> Pipeline:
    """Light cleanup before sending text to an LLM. LLMs read informal text fine and lose
    nuance when it is stripped, so this only masks links and tidies repeated punctuation."""
    return Pipeline([
        Remove(URL, placeholder="<URL>", name="mask_urls"),
        Remove(EMAIL, placeholder="<EMAIL>", name="mask_emails"),
        TidyPunct(),
    ], lexicon, name="llm")


def classic(lexicon: Optional[Lexicon] = None) -> Pipeline:
    """The textbook 'clean everything' pipeline, for comparison in class.
    It throws away negation, emoji and numbers: expect critical warnings."""
    return Pipeline([
        Remove(URL, EMAIL, MENTION, name="remove_links_mentions"),
        Remove(EMOJI, EMOTICON, name="remove_emoji"),
        Hashtags("remove"),
        RemovePunct(),
        Lowercase(),
        Elongation(),
        Slang(),
        Numbers("remove"),
        Stopwords(keep_negation=False),
        Stem(),
    ], lexicon, name="classic")


PRESETS = ("sentiment", "topic", "search", "stt_answer", "tts", "llm", "classic")
