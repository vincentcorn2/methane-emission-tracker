# Data leakage and independence audit

Three checks: site-level training overlap, temporal proximity between training and evaluation acquisitions at the same site, and conformal calibration set independence from candidate sites.

## (1) Candidate site overlap with training set

| Candidate | In training? | Role(s) | N crops |
|---|---|---|---|
| belchatow | yes | negative, positive, synthetic_substrate | 7 |
| rybnik | yes | positive | 2 |
| weisweiler | yes | positive | 1 |
| lippendorf | no | — | 0 |
| neurath | yes | negative | 1 |
| boxberg | yes | negative | 1 |
| groningen | yes | negative, positive, synthetic_substrate | 35 |
| maasvlakte | no | — | 0 |

## (2) Temporal proximity between evaluation and training dates

Same-site training crops within 14 days of an evaluation acquisition are flagged as potential leakage. Within-tile crops from other dates do not leak per-pixel labels.

| Site | Eval date | Nearest training crop | Days apart | Flag |
|---|---|---|---|---|
| belchatow | 2020-06-01 | belchatow_T34UCB_20240824 | 1545 | OK |
| belchatow | 2021-06-06 | belchatow_T34UCB_20240824 | 1175 | OK |
| belchatow | 2021-09-09 | belchatow_T34UCB_20240824 | 1080 | OK |
| belchatow | 2024-04-11 | belchatow_T34UCB_20240824 | 135 | OK |
| belchatow | 2024-05-26 | belchatow_T34UCB_20240824 | 90 | OK |
| belchatow | 2024-07-10 | belchatow_T34UCB_20240824 | 45 | OK |
| belchatow | 2024-07-30 | belchatow_T34UCB_20240824 | 25 | OK |
| belchatow | 2024-10-28 | belchatow_T34UCB_20240824 | 65 | OK |
| neurath | 2024-06-25 | neurath_T32ULB_20240920 | 87 | OK |
| neurath | 2024-08-29 | neurath_T32ULB_20240920 | 22 | OK |
| rybnik | 2025-03-22 | silesia_rybnik_T34UCA_20240628_enh19 | 267 | OK |

## (3) Conformal calibration set independence

Checked 28 OK-status calibration sites for proximity (< 50 km) to any candidate site.

**No conformal calibration sites within 50 km of any candidate site.** The threshold τ = 4.1052 is calibrated on a set that is spatially independent of the evaluation sites.


## (4) Threshold selection methodology

The conformal threshold τ = 4.1052 was computed by the split conformal prediction quantile on the non-emitter calibration set scores, without reference to the candidate-site backfill outcomes. The retraining hyperparameter selection (v1-v11) used a small held-out set of training crops (3 negatives) for validation loss monitoring, not the candidate-site evaluation outcomes. The candidate-site results were computed only after v8 was fixed and τ was calibrated.
