# rapikan

Explainable preprocessing for informal Indonesian text.

Most Indonesian NLP tutorials run the same pipeline: lowercase, strip everything that isn't a letter, remove stopwords, stem with Sastrawi. That pipeline turns **"Kuahnya nggak enak 😞"** into **"kuah enak"**. A negative review becomes positive, because "tidak" is on the stopword list and the emoji is gone.

rapikan is built around that problem:

- **Typed tokens.** `Rp28.000`, `3-0`, `8/10`, `19:30`, `payment-api`, URLs and emoji each stay one token and know what they are.
- **Negation is never dropped by default.** Stopword removal keeps `tidak`, `bukan`, `belum`, `jangan`. Optionally negation is merged into the word it negates (`tidak_enak`, `bukan_terbaik`).
- **Presets per task**, because the right amount of cleaning depends on the task: `sentiment`, `topic`, `search`, `stt_answer`, `tts`, `llm`, plus `classic` (the tutorial pipeline, for comparison).
- **Every run explains itself.** A per-step log of what was removed or changed, and warnings when meaning is lost.
- **Informal Indonesian rules.** Elongation (`enakkk`, `GOOLLL`), reduplication (`jelek2`), slang (`nggak`, `bgt`, `gpp`), colloquial verbs (`nyari` → `mencari`, `ngantri` → `mengantre`), hashtag splitting (`#kulinersolo` → `kuliner solo`), spoken fillers (`eee`, `anu`).
- **Its own stemmer.** A dictionary-checked affix search that uses the affix licensing in the hunspell-id data. It is on par with Sastrawi on standard words, better on informal ones, and roughly 60× faster.
- **Speech helpers.** Numbers and money to words for TTS (`Rp28.000` → "dua puluh delapan ribu rupiah"), and `match_answer` for checking a child's spoken quiz answer.
- **No runtime dependencies.** Pure Python 3.9+.

```python
from rapikan import presets

r = presets.sentiment()("Pelayanannya gak ramah, tp bakminya enak bgt 😍")
r.words
# ['pelayanannya', 'tidak_ramah', 'tapi', 'bakminya', 'enak', 'sekali', '😍']

r = presets.classic()("Kuahnya nggak enak 😞")
r.words      # ['kuah', 'enak']
print(r.report())
```
```
pipeline: classic
  + tokenize               4 tokens
  + remove_links_mentions  no change
  + remove_emoji           -1 removed
      x 😞
  ...
  + stopwords              -1 removed
      x tidak
  + stem                   1 changed
      kuahnya -> kuah
result: ['kuah', 'enak']
warnings:
  [critical] "nggak enak" lost its negation and now reads "enak".
  [warning] Emoji removed (😞). For sentiment, emoji are often the clearest emotion signal.
```

## Install

```bash
pip install -e .            # from a clone
pip install -e ".[dev,bench]"   # with pytest and Sastrawi for the benchmark
```

## Presets

| preset | for | keeps | removes / changes |
|---|---|---|---|
| `sentiment` | positive/negative with BoW or TF-IDF | emoji, negation (merged), intensifiers | links, mentions, numbers, punctuation; no stemming |
| `topic` | complaint categories, clustering, keyword counts | numbers, hashtags (split), negation | emoji, stopwords; stems |
| `search` | indexing and queries | numbers, hashtags (split) | emoji, mentions, stopwords incl. negation; stems |
| `stt_answer` | speech-to-text answers in a quiz | negation | fillers, punctuation; numbers → words; stems |
| `tts` | text read aloud | punctuation (tidied), case | links, emoji, mentions, hashtags; numbers and money → words; slang → formal |
| `llm` | before sending to an LLM | everything | masks URLs and emails, tidies `!!!` |
| `classic` | showing what goes wrong | | everything, like most tutorials |

Same text, three tasks:

```python
text = "MANTAPPP!!! 😍 Bakminya enakkk bgt, tp pelayanannya nggak ramah... Antriannya 45 menit, harga naik jd Rp28.000 #kulinersolo"

presets.sentiment()(text).words
# ['mantap', '😍', 'bakminya', 'enak', 'sekali', 'tapi', 'pelayanannya', 'tidak_ramah', 'antriannya',
#  'menit', 'harga', 'naik', 'jadi', 'kuliner', 'solo']

presets.topic()(text).words
# ['mantap', 'bakmi', 'enak', 'sekali', 'layan', 'tidak_ramah', 'antri', '45', 'menit', 'harga',
#  'naik', 'Rp28.000', 'kuliner', 'solo']

presets.tts()(text).text
# 'mantap! Bakminya enak sekali, tapi pelayanannya tidak ramah... Antriannya empat puluh lima menit,
#  harga naik jadi dua puluh delapan ribu rupiah'
```

## Build your own pipeline

```python
from rapikan import Pipeline, Remove, Hashtags, RemovePunct, Lowercase, Elongation, Slang, Numbers, Negation, Stopwords, Stem
from rapikan.tokens import URL, MENTION

p = Pipeline([
    Remove(URL, MENTION),
    Hashtags("split"),          # keep | strip | split | remove
    RemovePunct(keep=["?"]),
    Lowercase(),
    Elongation(),
    Slang(),
    Numbers("placeholder"),     # keep | remove | words | placeholder
    Negation("merge"),          # merge | tag
    Stopwords(keep_negation=True, extra=["sih"]),
    Stem(),
], name="mine")

p.describe()
p.batch(["text one", "text two"])   # list of word lists, ready for scikit-learn
```

Each step is a small class with a `run(tokens, ctx, log)` method, so writing your own is a few lines.

## Word lists

```python
from rapikan import Lexicon, presets

lex = (Lexicon.default()
       .with_slang({"mantul": "mantap betul"})
       .with_stopwords(add=["kak"], remove=["lagi"])
       .with_protected(["bakmi bejo", "kubernetes"]))   # never slang-mapped or stemmed
p = presets.topic(lex)
```

Bundled, and how they are licensed:

| list | size | source | license |
|---|---|---|---|
| root words + allowed affixes | 43k | hunspell-id (LibreOffice), processed by `tools/build_roots.py` | LGPL-3.0, commercial use allowed |
| slang → formal | ~400 | written for rapikan | MIT |
| stopwords | 161 | written for rapikan, negation and intensifiers left out on purpose | MIT |

Not bundled, but you can plug them in. Check their terms for your use:

- **Kateglo roots** (the list inside Sastrawi and nlp-id) is CC-BY-NC-SA 3.0, so non-commercial only. Use `Lexicon.default().with_sastrawi_roots()`.
- **Colloquial Indonesian Lexicon** (Salsabila et al., 2018, about 15k slang pairs) has no license file in its repository. Use `lex.with_slang_csv("colloquial-indonesian-lexicon.csv")`.

## Stemmer

```python
from rapikan import Stemmer
s = Stemmer()
s.stem("pelayanannya")        # 'layan'
s.stem("mengirim")            # 'kirim'
s.explain("mengirim")         # [('kirim', ('me',), (), 'licensed'), ('irim', ('me',), (), '1 unlicensed')]
```

It enumerates every decomposition the Indonesian affix grammar allows (up to three prefixes, derivational suffix, possessive, particle, with meN-/peN- sound changes), keeps those whose root is in the dictionary, and ranks them:

1. Prefer decompositions the dictionary licenses. hunspell-id records which affixes each root takes, so `mengirim` → *kirim* (takes meN-) beats *irim* (takes nothing), and `pembelian` → *beli* beats *belian*.
2. Then fewer affixes removed (`kebersihan` → *bersih*, not *sih*).
3. Then rule order (`memakan` → *makan* before *pakan*, `terasa` → *rasa* before *asa*).

Words already in the dictionary are left alone, so *mereka*, *kalian* and *sebagai* stay intact.

## Benchmark

`python bench/run_benchmark.py` compares against Sastrawi 1.0.1. The gold sets use standard KBBI roots and were written by hand for this project.

| set | words | rapikan | Sastrawi |
|---|---|---|---|
| **held-out** (written after tuning, run once) | 158 | **97.5%** | **96.8%** |
| development (used while tuning, so optimistic) | 221 | 97.7% | 92.8% |

Read the held-out row. On standard vocabulary the two stemmers are about equal: 4 errors vs 5 out of 158. The development set flatters rapikan because its rules were tuned on it.

Where rapikan does better:

- **ter-/ber-/pe- + r roots**: *terasa*, *terawat*, *berupa*, *perokok*, *perawat*. Sastrawi gives *asa*, *awat*, *upa*, *okok*, *awat*.
- **Unknown words with clitics**: *gamenya* → *game*, *errornya* → *error*. Sastrawi leaves them unchanged.
- **Speed**: 30k tokens in 9 ms vs 590 ms. Load time is 107 ms vs 16 ms, because rapikan loads the affix data.

Where it does worse, or both fail: lexicalised entries in the hunspell list stay unstemmed (*belajar*, *berangkat*, *tertawa*, *pelari*, *pemuda*), and *mempelajari* gives *pelajar* rather than *ajar*.

Meaning preservation on 8 sentiment sentences (negation and emoji signals):

| pipeline | signals kept |
|---|---|
| rapikan `sentiment` | 12 / 12 |
| tutorial pipeline (regex + Sastrawi stopwords + Sastrawi stemmer) | 5 / 12 |

Sastrawi's stopword list contains *tidak* and *belum*. Special tokens kept whole: rapikan 10/10, a `[a-z0-9]+` regex tokenizer 1/10.

Caveats: the gold sets are small (379 words in total) and written by one author, so treat the numbers as a sanity check rather than a published evaluation. Contributions of larger, independently labelled sets are welcome.

## Speech: TTS and spoken answers

```python
from rapikan import normalize_for_tts, match_answer, terbilang

normalize_for_tts("Harga naik jd Rp28.000!!! Skornya 8/10, jam 19:30, juara ke-2")
# 'Harga naik jadi dua puluh delapan ribu rupiah! Skornya delapan per sepuluh, jam sembilan belas tiga puluh, juara kedua'

m = match_answer("eee... kucingnyaaa! bukan anjing", "kucing")
m.correct, m.rejected          # (True, ['anjing'])
match_answer("bukan kucing", "kucing").correct   # False: the child negated it
m = match_answer("anjing... eh kucing", "kucing")
m.correct, m.ambiguous, m.others   # (True, True, ['anjing']): a self-correction the game may want to handle
```

`match_answer` puts both sides through the same `stt_answer` pipeline, treats words right after a negation as rejected, matches numbers as words ("2" = "dua"), and accepts near misses from the speech engine (`fuzzy=0.8`; set `1.0` to disable).

## Known limitations

- The slang list is small (~400) on purpose: every entry was checked by hand. Ambiguous words are left out (`kaya` can mean "rich" or "like", `buat` can mean "make" or "for", `kasih` would break "terima kasih").
- Hashtag splitting uses dictionary segmentation and prefers fewer, longer words. Unknown names stay joined.
- Negation scope is one content word. "tidak kenal lelah" becomes `tidak_kenal lelah`, and the double negative is not resolved.
- No part-of-speech tagging or lemmatisation beyond stemming.

## Development

```bash
pip install -e ".[dev,bench]"
pytest                          # unit tests and doctests
python bench/run_benchmark.py   # accuracy vs Sastrawi
python examples/demo_kuliah.py  # the four class test texts through every preset
```

## License

Code: MIT. Bundled root-word data (`src/rapikan/data/roots.tsv`): LGPL-3.0, derived from hunspell-id. See `src/rapikan/data/LICENSE-roots-LGPL-3.0.txt` and `NOTICE`.
