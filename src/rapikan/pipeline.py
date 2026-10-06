"""Explainable preprocessing pipeline.

A Pipeline is a list of steps. Running it returns a Result that keeps the
final tokens, a per-step log of what was removed or changed, and warnings
when information that usually matters was lost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from . import normalize as N
from .lexicon import Lexicon, resolve
from .numbers import ordinal, spell_token
from .stemmer import Stemmer
from .tokens import (EMAIL, EMOJI, EMOTICON, HASHTAG, MENTION, MONEY, NUMBER, PUNCT, URL, WORD, Token,
                     detokenize, tokenize)

CRITICAL, WARNING, INFO = "critical", "warning", "info"


@dataclass
class Warn:
    level: str
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.message}"


@dataclass
class StepLog:
    name: str
    active: bool = True
    removed: List[Token] = field(default_factory=list)
    changed: List[Tuple[str, str]] = field(default_factory=list)
    added: int = 0
    note: str = ""

    def summary(self) -> str:
        if not self.active:
            return "skipped"
        parts = []
        if self.removed:
            parts.append(f"-{len(self.removed)} removed")
        if self.changed:
            parts.append(f"{len(self.changed)} changed")
        if self.added:
            parts.append(f"+{self.added} added")
        return ", ".join(parts) or (self.note or "no change")


@dataclass
class Result:
    original: str
    tokens: List[Token]
    log: List[StepLog]
    warnings: List[Warn]
    pipeline: str = "custom"

    @property
    def words(self) -> List[str]:
        return [t.text for t in self.tokens]

    @property
    def text(self) -> str:
        return detokenize(self.tokens)

    def report(self, detail: bool = True) -> str:
        lines = [f"pipeline: {self.pipeline}"]
        width = max(len(s.name) for s in self.log) + 2
        for s in self.log:
            mark = "+" if s.active else "-"
            lines.append(f"  {mark} {s.name.ljust(width)}{s.summary()}")
            if detail and s.active:
                bits = [f"x {t.text}" for t in s.removed[:8]] + [f"{a} -> {b}" for a, b in s.changed[:8]]
                if bits:
                    more = len(s.removed) + len(s.changed) - len(bits)
                    lines.append("      " + " | ".join(bits) + (f" | ...+{more}" if more > 0 else ""))
        lines.append(f"result: {self.words}")
        if self.warnings:
            lines.append("warnings:")
            lines += [f"  {w}" for w in self.warnings]
        else:
            lines.append("warnings: none")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Result({self.words!r}, warnings={len(self.warnings)})"


class Context:
    def __init__(self, lexicon: Lexicon, stemmer: Stemmer):
        self.lex = lexicon
        self.stemmer = stemmer
        self.warnings: List[Warn] = []
        self.history: List[List[Token]] = []

    def is_word(self, w: str) -> bool:
        """Known word, or a derived form whose root is known (bermain -> main)."""
        if self.lex.is_known(w):
            return True
        root = self.stemmer.stem(w)
        return root != w and root in self.lex.roots and len(root) >= 3 and not w.startswith(root + "nya")

    def warn(self, level: str, code: str, message: str) -> None:
        self.warnings.append(Warn(level, code, message))


class Step:
    name = "step"

    def run(self, tokens: List[Token], ctx: Context, log: StepLog) -> List[Token]:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({params})"


# ---------------------------------------------------------------- steps

class Remove(Step):
    """Drop tokens of the given types (urls, mentions, emails, ...)."""

    _MESSAGES = {
        EMOJI: (WARNING, "emotion_lost", "Emoji removed ({items}). For sentiment, emoji are often the clearest emotion signal."),
        EMOTICON: (WARNING, "emotion_lost", "Emoticons removed ({items})."),
        HASHTAG: (INFO, "topic_lost", "Hashtags removed ({items}). They often mark the topic."),
        MONEY: (WARNING, "numbers_lost", "Amounts removed ({items}). Prices are often the core of a complaint."),
        NUMBER: (WARNING, "numbers_lost", "Numbers removed ({items}): scores, durations and counts can be the key fact."),
    }

    def __init__(self, *types: str, name: Optional[str] = None, placeholder: Optional[str] = None):
        self.types = types
        self.placeholder = placeholder
        self.name = name or "remove_" + "_".join(types)

    def run(self, tokens, ctx, log):
        out = []
        for t in tokens:
            if t.type in self.types:
                if self.placeholder:
                    out.append(t.copy(text=self.placeholder))
                    log.changed.append((t.text, self.placeholder))
                else:
                    log.removed.append(t)
            else:
                out.append(t)
        by_type = {}
        for t in log.removed:
            by_type.setdefault(t.type, []).append(t.text)
        for typ, items in by_type.items():
            if typ in self._MESSAGES:
                lvl, code, msg = self._MESSAGES[typ]
                ctx.warn(lvl, code, msg.format(items=" ".join(items[:6])))
        return out


class Hashtags(Step):
    """mode: 'keep', 'strip' (#kuliner -> kuliner), 'split' (#SemangatGaruda -> semangat garuda), 'remove'."""
    name = "hashtags"

    def __init__(self, mode: str = "split"):
        assert mode in ("keep", "strip", "split", "remove")
        self.mode = mode

    def run(self, tokens, ctx, log):
        if self.mode == "keep":
            log.note = "kept"
            return tokens
        if self.mode == "remove":
            return Remove(HASHTAG).run(tokens, ctx, log)
        out = []
        for t in tokens:
            if t.type != HASHTAG:
                out.append(t)
                continue
            parts = [t.text[1:]] if self.mode == "strip" else N.split_hashtag(t.text, ctx.lex, ctx.is_word)
            log.changed.append((t.text, " ".join(parts)))
            for p in parts:
                out.append(Token(p, WORD, orig=t.orig, start=t.start, end=t.end, tags={"from_hashtag"}))
        return out


class RemovePunct(Step):
    name = "remove_punct"

    def __init__(self, keep: Sequence[str] = ()):
        self.keep = tuple(keep)

    def run(self, tokens, ctx, log):
        out = []
        for t in tokens:
            if t.type == PUNCT and not any(k in t.text for k in self.keep):
                log.removed.append(t)
            else:
                out.append(t)
        if any("?" in t.text for t in log.removed):
            ctx.warn(INFO, "question_lost", "Question marks removed: questions and statements now look the same.")
        return out


class TidyPunct(Step):
    """Collapse '!!!' -> '!', '??' -> '?', '…' -> '...'. Keeps punctuation for TTS prosody."""
    name = "tidy_punct"

    def run(self, tokens, ctx, log):
        out = []
        for t in tokens:
            if t.type == PUNCT:
                new = t.text
                if new == "…" or (len(new) > 1 and set(new) == {"."}):
                    new = "..."
                elif len(new) > 1 and set(new) <= {"!", "?"}:
                    new = "?" if "?" in new else "!"
                if new != t.text:
                    log.changed.append((t.text, new))
                    t = t.copy(text=new)
                    t.tags.add("intense")
            out.append(t)
        return out


class Lowercase(Step):
    """mode 'all' lowercases every word; 'known_caps' only lowercases ALL-CAPS words that are
    real words (SALAH -> salah) and leaves acronyms (API, GBK) alone."""
    name = "lowercase"

    def __init__(self, mode: str = "all"):
        assert mode in ("all", "known_caps")
        self.mode = mode

    def run(self, tokens, ctx, log):
        out, emph = [], []
        for t in tokens:
            if t.type in (WORD, HASHTAG):
                low = t.text.lower()
                do = self.mode == "all" or ("caps" in t.tags and ctx.lex.is_known(N._RUN3.sub(r"\1", low)))
                if do and low != t.text:
                    if "caps" in t.tags and len(t.text) > 2:
                        emph.append(t.text)
                    log.changed.append((t.text, low))
                    t = t.copy(text=low)
            out.append(t)
        if emph and self.mode == "all":
            ctx.warn(INFO, "emphasis_lost",
                     f"ALL-CAPS emphasis lowercased ({' '.join(emph[:5])}). Tokens keep a 'caps' tag if you need it.")
        return out


class _WordMap(Step):
    """Base for steps that rewrite one WORD into zero or more words."""
    types = (WORD,)

    def rewrite(self, t: Token, ctx: Context) -> Optional[List[str]]:  # pragma: no cover
        raise NotImplementedError

    def tag(self) -> str:
        return self.name

    def run(self, tokens, ctx, log):
        out = []
        for t in tokens:
            if t.type not in self.types or t.text.lower() in ctx.lex.protected:
                out.append(t)
                continue
            new = self.rewrite(t, ctx)
            if new is None or new == [t.text]:
                out.append(t)
                continue
            log.changed.append((t.text, " ".join(new)))
            if len(new) > 1:
                log.added += len(new) - 1
            for i, w in enumerate(new):
                nt = t.copy(text=_match_case(w, t.text) if i == 0 else w)
                nt.tags.add(self.tag())
                out.append(nt)
        return out


def _match_case(new: str, old: str) -> str:
    if old.isupper() and len(old) > 1:
        return new.upper()
    if old[:1].isupper() and old[1:].islower():
        return new[:1].upper() + new[1:]
    return new


class Elongation(_WordMap):
    """'enakkk' -> 'enak', 'GOOLLL' -> 'GOL'. Tags the token 'elongated' (an intensity signal)."""
    name = "elongation"

    def rewrite(self, t, ctx):
        r = N.collapse_elongation(t.text, ctx.lex)
        return [r] if r else None

    def tag(self):
        return "elongated"


class Reduplication(_WordMap):
    """'jelek2' -> 'jelek-jelek', 'anak2nya' -> 'anak-anaknya'."""
    name = "reduplication"

    def rewrite(self, t, ctx):
        r = N.expand_reduplication(t.text)
        return [r] if r else None


class Slang(_WordMap):
    """Informal -> formal ('nggak' -> 'tidak', 'gpp' -> 'tidak apa-apa').
    With colloquial_verbs=True also 'nyari' -> 'mencari', 'ngantri' -> 'mengantri'."""
    name = "slang"

    def __init__(self, colloquial_verbs: bool = True):
        self.colloquial_verbs = colloquial_verbs

    def rewrite(self, t, ctx):
        low = t.text.lower()
        if low in ctx.lex.slang:
            return ctx.lex.slang[low].split()
        if self.colloquial_verbs and "technical" not in t.tags:
            f = N.formal_from_colloquial(low, ctx.lex)
            if f:
                return [f]
        return None


class Fillers(Step):
    """Drop spoken fillers (eee, ehm, anu, hmm). Mostly for speech-to-text output."""
    name = "fillers"

    def run(self, tokens, ctx, log):
        out = []
        for t in tokens:
            if t.type == WORD and N._RUN3.sub(r"\1", t.text.lower()) in ctx.lex.fillers:
                log.removed.append(t)
            else:
                out.append(t)
        return out


class Numbers(Step):
    """mode: 'keep', 'remove', 'words' (Rp28.000 -> dua puluh delapan ribu rupiah), or 'placeholder' (<NUM>)."""
    name = "numbers"

    def __init__(self, mode: str = "keep", placeholder: str = "<NUM>"):
        assert mode in ("keep", "remove", "words", "placeholder")
        self.mode = mode
        self.placeholder = placeholder

    def run(self, tokens, ctx, log):
        if self.mode == "keep":
            log.note = "kept"
            return tokens
        if self.mode == "remove":
            return Remove(NUMBER, MONEY).run(tokens, ctx, log)
        out = []
        for t in tokens:
            spoken = None
            if t.type in (NUMBER, MONEY):
                spoken = self.placeholder if self.mode == "placeholder" else spell_token(t.text)
            elif t.type == WORD and self.mode == "words":
                spoken = ordinal(t.text)
            if not spoken:
                out.append(t)
                continue
            log.changed.append((t.text, spoken))
            words = [spoken] if self.mode == "placeholder" else spoken.split()
            log.added += len(words) - 1
            for w in words:
                out.append(Token(w, WORD, orig=t.orig, start=t.start, end=t.end, tags={"from_number"}))
        return out


class Negation(Step):
    """mode 'merge': 'tidak enak' -> 'tidak_enak' (one feature for bag-of-words models).
    mode 'tag': keeps both tokens and tags the next word 'negated'."""
    name = "negation"

    def __init__(self, mode: str = "merge"):
        assert mode in ("merge", "tag")
        self.mode = mode

    def run(self, tokens, ctx, log):
        out: List[Token] = []
        neg, stop = ctx.lex.negations, ctx.lex.stopwords
        i = 0
        while i < len(tokens):
            t = tokens[i]
            j = self._target(tokens, i, neg, stop) if t.type == WORD and t.text.lower() in neg else None
            if j is None:
                out.append(t)
                i += 1
                continue
            nxt = tokens[j]
            skipped = tokens[i + 1:j]  # function words between: 'bukan yang terbaik'
            if self.mode == "merge":
                merged = Token(f"{t.text}_{nxt.text}", WORD, orig=f"{t.orig} {nxt.orig}",
                               start=t.start, end=nxt.end, tags=t.tags | nxt.tags | {"negated"})
                log.changed.append((" ".join(x.text for x in tokens[i:j + 1]), merged.text))
                out.extend(skipped)
                out.append(merged)
            else:
                tagged = nxt.copy()
                tagged.tags.add("negated")
                log.changed.append((nxt.text, nxt.text + " [negated]"))
                out.append(t)
                out.extend(skipped)
                out.append(tagged)
            i = j + 1
        return out

    @staticmethod
    def _target(tokens, i, neg, stop, max_skip: int = 2):
        """Index of the content word a negation applies to, skipping up to two function words."""
        for j in range(i + 1, min(len(tokens), i + 2 + max_skip)):
            tj = tokens[j]
            if tj.type != WORD or tj.text.lower() in neg:
                return None
            if tj.text.lower() not in stop:
                return j
        return None


class Stopwords(Step):
    """Remove function words. Negation is kept unless keep_negation=False."""
    name = "stopwords"

    def __init__(self, keep_negation: bool = True, extra: Iterable[str] = (), keep: Iterable[str] = ()):
        self.keep_negation = keep_negation
        self.extra = frozenset(w.lower() for w in extra)
        self.keep = frozenset(w.lower() for w in keep)

    def run(self, tokens, ctx, log):
        stop = (ctx.lex.stopwords | self.extra) - self.keep
        if not self.keep_negation:
            stop = stop | ctx.lex.negations
        out = []
        for i, t in enumerate(tokens):
            low = t.text.lower()
            if t.type == WORD and low in stop:
                log.removed.append(t)
                if low in ctx.lex.negations:
                    nxt = tokens[i + 1] if i + 1 < len(tokens) else None
                    if nxt is not None and nxt.type == WORD:
                        ctx.warn(CRITICAL, "negation_lost",
                                 f'"{t.orig} {nxt.orig}" lost its negation and now reads "{nxt.text}".')
            else:
                out.append(t)
        return out


class Stem(Step):
    """Reduce words to their root. Technical tokens and protected words are left alone."""
    name = "stem"

    def run(self, tokens, ctx, log):
        out = []
        for t in tokens:
            if t.type != WORD or "technical" in t.tags or "from_number" in t.tags:
                out.append(t)
                continue
            if "negated" in t.tags and "_" in t.text:
                neg, _, word = t.text.partition("_")
                new = f"{neg}_{ctx.stemmer.stem(word)}"
            else:
                new = ctx.stemmer.stem(t.text)
                if t.text[:1].isupper() and new == t.text.lower():
                    new = t.text  # untouched proper noun
            if new != t.text:
                log.changed.append((t.text, new))
                t = t.copy(text=new)
            out.append(t)
        return out


# --------------------------------------------------------------- pipeline

class Pipeline:
    def __init__(self, steps: Sequence[Step], lexicon: Optional[Lexicon] = None,
                 stemmer: Optional[Stemmer] = None, name: str = "custom"):
        self.steps = list(steps)
        self.lexicon = resolve(lexicon)
        self.stemmer = stemmer or Stemmer(self.lexicon)
        self.name = name

    def __call__(self, text: str) -> Result:
        return self.run(text)

    def run(self, text: str) -> Result:
        ctx = Context(self.lexicon, self.stemmer)
        tokens = tokenize(text)
        log = [StepLog("tokenize", note=f"{len(tokens)} tokens")]
        for step in self.steps:
            entry = StepLog(step.name)
            tokens = step.run(tokens, ctx, entry)
            log.append(entry)
        return Result(text, tokens, log, _dedupe(ctx.warnings), self.name)

    def batch(self, texts: Iterable[str]) -> List[List[str]]:
        """Words only, for feeding scikit-learn and friends."""
        return [self.run(t).words for t in texts]

    def describe(self) -> str:
        return f"{self.name}:\n" + "\n".join(f"  {i + 1}. {s!r}" for i, s in enumerate(self.steps))

    def __repr__(self) -> str:
        return f"Pipeline({self.name!r}, {len(self.steps)} steps)"


def _dedupe(warns: List[Warn]) -> List[Warn]:
    seen, out = set(), []
    for w in warns:
        if (w.code, w.message) not in seen:
            seen.add((w.code, w.message))
            out.append(w)
    order = {CRITICAL: 0, WARNING: 1, INFO: 2}
    return sorted(out, key=lambda w: order[w.level])


def process(text: str, preset: str = "topic") -> Result:
    from . import presets
    return getattr(presets, preset)()(text)
