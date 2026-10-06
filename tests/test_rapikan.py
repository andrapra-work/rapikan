import pytest

import rapikan
from rapikan import Lexicon, Pipeline, Stemmer, match_answer, normalize_for_tts, presets, terbilang, tokenize
from rapikan.normalize import collapse_elongation, expand_reduplication, men_prefix, split_hashtag
from rapikan.lexicon import default_lexicon

LEX = default_lexicon()


# ------------------------------------------------------------- tokenizer

@pytest.mark.parametrize("text,token,typ", [
    ("Harga naik jd Rp28.000, mahal", "Rp28.000", "money"),
    ("tiket 150rb aja", "150rb", "money"),
    ("Menang 3-0!!!", "3-0", "number"),
    ("Skornya 8/10", "8/10", "number"),
    ("restart 15x", "15x", "number"),
    ("jam 19:30", "19:30", "number"),
    ("naik 10%", "10%", "number"),
    ("Timnas U-23", "U-23", "word"),
    ("pod payment-api", "payment-api", "word"),
    ("cek https://grafana.internal/d/pay.", "https://grafana.internal/d/pay", "url"),
    ("halo @bakmiayambejo", "@bakmiayambejo", "mention"),
    ("#kulinersolo", "#kulinersolo", "hashtag"),
    ("mantap 😍", "😍", "emoji"),
    ("Indonesia 🇮🇩", "🇮🇩", "emoji"),
    ("enak :)", ":)", "emoticon"),
    ("email ke andra@contoh.id ya", "andra@contoh.id", "email"),
])
def test_tokens_stay_whole(text, token, typ):
    toks = {t.text: t.type for t in tokenize(text)}
    assert toks.get(token) == typ, toks


def test_tokenizer_tags():
    toks = {t.text: t for t in tokenize("ENAK OOMKilled jelek2 anak-anak payment-api")}
    assert "caps" in toks["ENAK"].tags
    assert "technical" in toks["OOMKilled"].tags
    assert "technical" not in toks["jelek2"].tags
    assert "compound" in toks["payment-api"].tags
    assert "compound" not in toks["anak-anak"].tags


def test_clean_text_handles_html_and_zero_width():
    assert rapikan.clean_text("enak&amp;murah​  banget") == "enak&murah banget"


# ------------------------------------------------------------ normalize

@pytest.mark.parametrize("word,expected", [
    ("enakkk", "enak"), ("kerennn", "keren"), ("GOOLLL", "GOL"), ("Kecewaaa", "Kecewa"),
])
def test_elongation(word, expected):
    assert collapse_elongation(word, LEX) == expected


@pytest.mark.parametrize("word", ["saat", "taat", "maaf", "enak"])
def test_elongation_keeps_real_doubles(word):
    assert collapse_elongation(word, LEX) is None


def test_reduplication():
    assert expand_reduplication("jelek2") == "jelek-jelek"
    assert expand_reduplication("anak2nya") == "anak-anaknya"
    assert expand_reduplication("U23") is None


@pytest.mark.parametrize("root,verb", [
    ("tunggu", "menunggu"), ("cari", "mencari"), ("pakai", "memakai"), ("cat", "mengecat"),
    ("kirim", "mengirim"), ("sapu", "menyapu"), ("baca", "membaca"), ("lihat", "melihat"),
    ("ambil", "mengambil"), ("gambar", "menggambar"),
])
def test_men_prefix(root, verb):
    assert men_prefix(root) == verb


def test_hashtag_split():
    assert split_hashtag("#SemangatGaruda", LEX) == ["semangat", "garuda"]
    assert split_hashtag("#kulinersolo", LEX) == ["kuliner", "solo"]
    assert split_hashtag("#xyzqq", LEX) == ["xyzqq"]


def test_slang_and_colloquial_verbs():
    p = Pipeline([rapikan.Lowercase(), rapikan.Slang()])
    assert p("nggak bgt").words == ["tidak", "sekali"]
    assert p("gpp kok").words == ["tidak", "apa-apa", "kok"]
    assert p("lagi nyari bakmi").words == ["lagi", "mencari", "bakmi"]
    assert p("ngantri lama").words == ["mengantre", "lama"]


# --------------------------------------------------------------- stemmer

@pytest.mark.parametrize("word,root", [
    ("makanan", "makan"), ("memakan", "makan"), ("mengirim", "kirim"), ("pengalaman", "alam"),
    ("pembelian", "beli"), ("kebersihan", "bersih"), ("terasa", "rasa"), ("berupa", "rupa"),
    ("diperbaiki", "baik"), ("pelayanannya", "layan"), ("anak-anaknya", "anak"), ("bermain-main", "main"),
    ("mereka", "mereka"), ("kalian", "kalian"), ("sebagai", "sebagai"), ("berita", "berita"),
    ("gamenya", "game"), ("errornya", "error"),
])
def test_stemmer(word, root):
    assert Stemmer().stem(word) == root


def test_stemmer_explain_lists_licensing():
    best = Stemmer().explain("mengirim")[0]
    assert best[0] == "kirim" and best[3] == "licensed"


def test_protected_words():
    lex = LEX.with_protected(["pelari"])
    assert Stemmer(lex).stem("pelari") == "pelari"


# --------------------------------------------------------------- presets

def test_sentiment_keeps_negation_and_emoji():
    r = presets.sentiment()("Kuahnya nggak enak 😞")
    assert "tidak_enak" in r.words and "😞" in r.words
    assert not [w for w in r.warnings if w.level == "critical"]


def test_classic_pipeline_warns_about_negation():
    r = presets.classic()("Kuahnya nggak enak 😞")
    assert "enak" in r.words and "tidak" not in r.words
    codes = {w.code for w in r.warnings}
    assert {"negation_lost", "emotion_lost"} <= codes


def test_topic_compresses_but_keeps_numbers():
    r = presets.topic()("Antriannya 45 menit, harga naik jd Rp28.000 #kulinersolo")
    assert r.words == ["antri", "45", "menit", "harga", "naik", "Rp28.000", "kuliner", "solo"]


def test_stopwords_never_drop_negation_by_default():
    p = Pipeline([rapikan.Lowercase(), rapikan.Slang(), rapikan.Stopwords()])
    assert "tidak" in p("ga suka sama sekali").words


def test_negation_tag_mode():
    p = Pipeline([rapikan.Lowercase(), rapikan.Negation("tag")])
    toks = p("tidak enak").tokens
    assert [t.text for t in toks] == ["tidak", "enak"] and "negated" in toks[1].tags


def test_report_is_readable():
    rep = presets.topic()("Pelayanannya nggak ramah 😤").report()
    assert "pipeline: topic" in rep and "stem" in rep and "result:" in rep


def test_batch():
    assert presets.topic().batch(["Bakminya enak", "Kuahnya asin"]) == [["bakmi", "enak"], ["kuah", "asin"]]


def test_tech_tokens_survive_topic():
    r = presets.topic()("Pod payment-api OOMKilled restart 15x")
    assert {"payment-api", "oomkilled", "15x"} <= set(r.words)


# ------------------------------------------------------------- numbers/TTS

@pytest.mark.parametrize("n,words", [
    (0, "nol"), (11, "sebelas"), (19, "sembilan belas"), (100, "seratus"), (1000, "seribu"),
    (1105, "seribu seratus lima"), (28000, "dua puluh delapan ribu"), (70000, "tujuh puluh ribu"),
    (1_500_000, "satu juta lima ratus ribu"), (2026, "dua ribu dua puluh enam"),
])
def test_terbilang(n, words):
    assert terbilang(n) == words


def test_tts():
    out = normalize_for_tts("Harga naik jd Rp28.000!!! 😤 Skornya 8/10, jam 19:30, juara ke-2")
    assert out == ("Harga naik jadi dua puluh delapan ribu rupiah! Skornya delapan per sepuluh, "
                   "jam sembilan belas tiga puluh, juara kedua")


# ------------------------------------------------------------- answers

@pytest.mark.parametrize("spoken,expected,correct", [
    ("eee... kucingnyaaa! bukan anjing", "kucing", True),
    ("bukan kucing", "kucing", False),
    ("ehm jerapahhh", "jerapah", True),
    ("ada 2", "dua", True),
    ("dua", "2", True),
    ("gajah", "jerapah", False),
    ("kucinf", "kucing", True),   # near miss from the STT engine
])
def test_match_answer(spoken, expected, correct):
    assert match_answer(spoken, expected).correct is correct


def test_match_answer_flags_self_correction():
    m = match_answer("anjing... eh kucing", "kucing")
    assert m.correct and m.ambiguous and m.others == ["anjing"]


# --------------------------------------------------------------- lexicon

def test_custom_slang_and_stopwords():
    lex = LEX.with_slang({"mantul": "mantap betul"}).with_stopwords(add=["bakmi"])
    p = presets.topic(lex)
    assert p("bakmi mantul").words == ["mantap", "betul"]
