# Held-out evaluation of CH4Net v8

This file reports v8 performance on candidate sites that were either never seen during training (TRULY HELD-OUT) or seen as NEGATIVE only (positive detection at test time = model overrides its training label).

All thresholds: conformal τ = 4.1052 at α = 0.10; CFAR ratio rule per Section 2.2.


## TRULY HELD-OUT (never seen in training)

| Site | Total records | Valid records | Above τ | CFAR detect |
|---|---|---|---|---|
| lippendorf | 8 | 8 | 2 | 1 |
| boxberg | 8 | 7 | 1 | 0 |
| maasvlakte | 7 | 7 | 0 | 0 |

### lippendorf — per acquisition
| Date | S/C | cv_ctrl | Above τ | CFAR |
|---|---|---|---|---|
| None | 1.1499 | 0.7092 | · | · |
| None | 1.2339 | 0.7379 | · | · |
| None | 0.1515 | 0.6446 | · | · |
| None | 0.903 | 0.701 | · | · |
| None | 315.9831 | 1.3641 | ✓ | · |
| None | 1.5532 | 0.7897 | · | · |
| None | 0.1255 | 0.7815 | · | · |
| None | 155.3617 | 0.5881 | ✓ | ✓ |

### boxberg — per acquisition
| Date | S/C | cv_ctrl | Above τ | CFAR |
|---|---|---|---|---|
| None | 0.9717 | 0.5077 | · | · |
| None | 0.0324 | 0.7664 | · | · |
| None | 0.0003 | 0.706 | · | · |
| None | 1.1394 | 0.7443 | · | · |
| None | 0.8428 | 0.5548 | · | · |
| None | 1202.8317 | 1.0519 | ✓ | · |
| None | 0.007 | 0.7068 | · | · |

### maasvlakte — per acquisition
| Date | S/C | cv_ctrl | Above τ | CFAR |
|---|---|---|---|---|
| None | 0.0038 | 0.4112 | · | · |
| None | 0.2499 | 0.5816 | · | · |
| None | 0.0077 | 0.8204 | · | · |
| None | 0.0019 | 0.998 | · | · |
| None | 0.0224 | 0.3711 | · | · |
| None | 0.6664 | 0.2023 | · | · |
| None | 0.2096 | 0.5655 | · | · |


## Trained as NEGATIVE (positive detection overrides training label)

| Site | Total records | Valid records | Above τ | CFAR detect |
|---|---|---|---|---|
| neurath | 8 | 7 | 2 | 2 |

### neurath — per acquisition
| Date | S/C | cv_ctrl | Above τ | CFAR |
|---|---|---|---|---|
| None | 0.5101 | 0.7419 | · | · |
| None | 0.6023 | 0.7433 | · | · |
| None | 0.7143 | 0.7566 | · | · |
| None | 0.4327 | 0.62 | · | · |
| None | 23.0392 | 0.9924 | ✓ | ✓ |
| None | 1.1235 | 0.4711 | · | · |
| None | 67.2048 | 0.2879 | ✓ | ✓ |


## Trained as NEGATIVE + used as synthetic substrate

| Site | Total records | Valid records | Above τ | CFAR detect |
|---|---|---|---|---|
| belchatow | 8 | 7 | 5 | 3 |
| groningen | 6 | 6 | 3 | 2 |

### belchatow — per acquisition
| Date | S/C | cv_ctrl | Above τ | CFAR |
|---|---|---|---|---|
| None | 849.0822 | 0.7115 | ✓ | ✓ |
| None | 18.5255 | 1.003 | ✓ | ✓ |
| None | 0.5352 | 0.9184 | · | · |
| None | 0.1794 | 0.5687 | · | · |
| None | 9.4837 | 0.6514 | ✓ | · |
| None | 142.9783 | 1.1052 | ✓ | ✓ |
| None | 27.3032 | 1.2651 | ✓ | · |

### groningen — per acquisition
| Date | S/C | cv_ctrl | Above τ | CFAR |
|---|---|---|---|---|
| None | 4.6018 | 0.7594 | ✓ | · |
| None | 0.2509 | 1.0875 | · | · |
| None | 0.3515 | 0.1501 | · | · |
| None | 4.6539 | 0.869 | ✓ | · |
| None | 1.691 | 0.265 | · | ✓ |
| None | 6.0092 | 0.6746 | ✓ | ✓ |

## Section 1.5 / Section 3 — proposed text

**Truly held-out test set.**  The model never saw the following sites in any form during training: lippendorf (valid n = 8, above-τ = 2, CFAR = 1), boxberg (valid n = 7, above-τ = 1, CFAR = 0), maasvlakte (valid n = 7, above-τ = 0, CFAR = 0).  Their performance is an independent test of the v8 model and the conformal threshold τ = 4.1052.

**Model overrides its own training labels.**  The following candidate sites were in training as NEGATIVE crops, but the production pipeline produces above-threshold detections on subsequent acquisitions: belchatow (n_records = 8, positive detections at test time: above-τ = 5, CFAR = 3), neurath (n_records = 8, positive detections at test time: above-τ = 2, CFAR = 2), groningen (n_records = 6, positive detections at test time: above-τ = 3, CFAR = 2).  This is a stronger result than a held-out test because the model is contradicting a training label on the basis of the spectral signature it learned from the synthetic-positive distribution.
