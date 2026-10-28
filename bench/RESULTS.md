# Benchmark results

Generated 2026-10-28 with Python 3.11, scikit-learn 1.8.0, PySastrawi 1.2.1, mpstemmer (git master), nlp-id 0.1.23.0.
Rerun: `python bench/compare_stemmers.py` and `python bench/downstream.py --smsa DIR --emot DIR --nusax DIR`.

# Stemming

## UD CSUI affixed (independent) (783 words)

| stemmer | accuracy | 95% CI | errors | vs rapikan (McNemar p) | time |
|---|---|---|---|---|---|
| rapikan | 97.8% | 96.8-98.7% | 17 | - | 14 ms |
| Sastrawi | 96.3% | 94.9-97.6% | 29 | p=0.023 (rapikan-only right 18, Sastrawi-only right 6) | 162 ms |
| mpstemmer | 90.4% | 88.4-92.5% | 75 | p=0.000 (rapikan-only right 60, mpstemmer-only right 2) | 3 ms |
| nlp-id | 98.0% | 96.9-98.9% | 16 | p=1.000 (rapikan-only right 7, nlp-id-only right 8) | 2 ms |

## rapikan held-out (158 words)

| stemmer | accuracy | 95% CI | errors | vs rapikan (McNemar p) | time |
|---|---|---|---|---|---|
| rapikan | 97.5% | 94.9-99.4% | 4 | - | 2 ms |
| Sastrawi | 96.8% | 93.7-99.4% | 5 | p=1.000 (rapikan-only right 3, Sastrawi-only right 2) | 27 ms |
| mpstemmer | 86.1% | 80.4-91.1% | 22 | p=0.000 (rapikan-only right 18, mpstemmer-only right 0) | 0 ms |
| nlp-id | 98.1% | 95.6-100.0% | 3 | p=1.000 (rapikan-only right 1, nlp-id-only right 2) | 1 ms |

## rapikan development (tuned on) (221 words)

| stemmer | accuracy | 95% CI | errors | vs rapikan (McNemar p) | time |
|---|---|---|---|---|---|
| rapikan | 97.7% | 95.5-99.5% | 5 | - | 2 ms |
| Sastrawi | 92.8% | 89.1-95.9% | 16 | p=0.007 (rapikan-only right 13, Sastrawi-only right 2) | 32 ms |
| mpstemmer | 81.0% | 75.6-86.0% | 42 | p=0.000 (rapikan-only right 39, mpstemmer-only right 2) | 1 ms |
| nlp-id | 96.8% | 94.6-99.1% | 7 | p=0.688 (rapikan-only right 4, nlp-id-only right 2) | 1 ms |

# Downstream classification (TF-IDF + logistic regression)

## SmSA sentiment (pre-cleaned text) (train 11000, eval 1260, 3 classes)

| pipeline | macro-F1 | 95% CI | vs minimal | vs tutorial |
|---|---|---|---|---|
| none (whitespace) | 0.845 | 0.818-0.870 | -0.004 [-0.018, +0.008] | +0.051 [+0.026, +0.076] * |
| minimal (lower+regex) | 0.850 | 0.823-0.875 | - | +0.055 [+0.030, +0.081] * |
| tutorial (Sastrawi) | 0.795 | 0.764-0.824 | -0.055 [-0.081, -0.030] * | - |
| rapikan classic | 0.816 | 0.786-0.843 | -0.034 [-0.057, -0.011] * | +0.021 [+0.001, +0.040] * |
| rapikan sentiment | 0.849 | 0.820-0.873 | -0.001 [-0.019, +0.015] | +0.054 [+0.027, +0.079] * |
| rapikan topic | 0.839 | 0.812-0.865 | -0.010 [-0.032, +0.012] | +0.045 [+0.025, +0.065] * |

`*` = 95% interval of the difference excludes zero.

## EmoT emotion (raw tweets) (train 3521, eval 440, 5 classes)

| pipeline | macro-F1 | 95% CI | vs minimal | vs tutorial |
|---|---|---|---|---|
| none (whitespace) | 0.606 | 0.557-0.649 | -0.053 [-0.089, -0.016] * | -0.016 [-0.057, +0.030] |
| minimal (lower+regex) | 0.659 | 0.611-0.701 | - | +0.037 [+0.002, +0.074] * |
| tutorial (Sastrawi) | 0.622 | 0.575-0.662 | -0.037 [-0.074, -0.002] * | - |
| rapikan classic | 0.650 | 0.604-0.691 | -0.009 [-0.048, +0.030] | +0.028 [-0.005, +0.064] |
| rapikan sentiment | 0.668 | 0.619-0.710 | +0.009 [-0.021, +0.039] | +0.046 [+0.012, +0.082] * |
| rapikan topic | 0.658 | 0.613-0.700 | -0.001 [-0.038, +0.040] | +0.036 [+0.006, +0.068] * |

`*` = 95% interval of the difference excludes zero.

## NusaX-senti Indonesian (original casing) (train 500, eval 400, 3 classes)

| pipeline | macro-F1 | 95% CI | vs minimal | vs tutorial |
|---|---|---|---|---|
| none (whitespace) | 0.697 | 0.649-0.743 | -0.039 [-0.073, -0.005] * | +0.006 [-0.051, +0.063] |
| minimal (lower+regex) | 0.736 | 0.688-0.779 | - | +0.046 [-0.004, +0.094] |
| tutorial (Sastrawi) | 0.691 | 0.640-0.734 | -0.046 [-0.094, +0.004] | - |
| rapikan classic | 0.683 | 0.635-0.727 | -0.054 [-0.094, -0.011] * | -0.008 [-0.050, +0.033] |
| rapikan sentiment | 0.732 | 0.683-0.776 | -0.004 [-0.031, +0.021] | +0.041 [-0.010, +0.095] |
| rapikan topic | 0.691 | 0.641-0.736 | -0.046 [-0.091, -0.006] * | -0.000 [-0.043, +0.042] |

`*` = 95% interval of the difference excludes zero.
