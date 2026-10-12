# Aegis-ML Robustness Report

Generated 2026-10-12T20:47:25.518772+00:00 (seed 1337).

Dataset: sklearn load_digits - 1167 train / 270 val / 360 test, 10 classes.

## Evasion (test accuracy under attack)

| eps | 0.05 | 0.1 | 0.2 | 0.3 |
| --- | --- | --- | --- | --- |
| FGSM standard | 0.881 | 0.611 | 0.039 | 0.000 |
| FGSM adv-trained | 0.953 | 0.875 | 0.614 | 0.228 |
| PGD standard | 0.875 | 0.561 | 0.014 | 0.000 |
| PGD adv-trained | 0.953 | 0.858 | 0.508 | 0.047 |

Baseline clean accuracy: 0.983 (standard), 0.964 (adversarial).

## Feature squeezing (adversarial input detection)

At eps=0.2, threshold calibrated to clean FPR 0.067: FGSM detection 0.519, PGD detection 0.472.

## Data poisoning

- Label flip (rate 0.3): accuracy 0.972 -> 0.950 poisoned; detector precision 0.897, recall 0.997; accuracy after removal 0.958.
- Backdoor (poison rate 0.06): clean accuracy 0.964, ASR 0.985; spectral detector precision 0.690, recall 0.986; ASR after filtering 0.015.

## Model extraction

| query budget | surrogate accuracy | fidelity |
| --- | --- | --- |
| 250 | 0.831 | 0.836 |
| 500 | 0.900 | 0.911 |
| 1000 | 0.928 | 0.931 |
| 2000 | 0.958 | 0.975 |

PRADA flags the extraction client after 176 queries; benign client flagged: False.
