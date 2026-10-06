"""The four test texts from the NLP class, run through every preset.

    python examples/demo_kuliah.py           # words per preset
    python examples/demo_kuliah.py --report  # full step-by-step report for 'classic'
"""
import sys

from rapikan import presets

TEXTS = {
    "Ulasan bakmi": "MANTAPPP!!! 😍🔥 Bakminya enakkk bgt, tp pelayanannya nggak ramah sama sekali... "
                    "Antriannya 45 menit, harga naik jd Rp28.000 😤 @bakmiayambejo tolong diperbaiki ya! "
                    "#kulinersolo https://maps.app.goo.gl/bejo123",
    "Komentar Timnas": "GOOLLL!!! 😭🇮🇩 Timnas menang 2-1 lawan Vietnam, pemainnya nggak kenal lelah bgt... "
                       "Penonton di GBK sampai 70.000 orang 🔥 Nonton highlight di https://youtu.be/abc123 "
                       "#TimnasDay @timnasindonesia kerennn!!!",
    "Alert ops": "[CRITICAL] Pod payment-api RESTARTTT 15x dlm 10 menit!!! 😱 Errornya nggak muncul di log, "
                 "cuma OOMKilled... Dashboard: https://grafana.internal/d/payment @oncall-sre tolong dicek "
                 "sekarang ya #incident #prod",
    "Chat game anak": 'Anakku main gamenya seneng bgt!!! 😍 Tadi jawab "kucingnyaaa" trs dibilang SALAH, '
                      "pdhl gambarnya beneran kucing 😅 Skornya 8/10, nggak jelek kan? @timgameanak tolong "
                      "dicek dong https://contoh.id/skor-anak #belajarsambilbermain",
}

if __name__ == "__main__":
    for title, text in TEXTS.items():
        print(f"\n=== {title}\n{text}\n")
        for name in ("classic", "sentiment", "topic", "tts"):
            r = getattr(presets, name)()(text)
            out = r.text if name == "tts" else " ".join(r.words)
            crit = sum(w.level == "critical" for w in r.warnings)
            print(f"{name:>9}: {out}")
            if crit:
                print(f"{'':>11}{crit} critical warning(s): " + "; ".join(w.message for w in r.warnings if w.level == "critical"))
        if "--report" in sys.argv:
            print()
            print(presets.classic()(text).report())
