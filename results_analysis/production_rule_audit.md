# Production detection rule audit

**Production rule (single source of truth):**  

A record is a DETECTION iff status=`ok` AND `sc_cfar > τ` AND `cfar_detect = True`.

- τ (α=0.10) = **3.5796** (read from `results_analysis/calibrated_threshold.json`)
- All other above-threshold outcomes are classified as below.

## Classification key

- **`DETECTION`** — passes the full production rule: status=ok AND sc_cfar > τ AND cfar_detect=True
- **`CFAR_SUPPRESSED`** — sc_cfar > τ but cfar_detect=False — high control heterogeneity inflated the local CFAR threshold above the observed S/C. NOT a production detection.
- **`SUB_TAU`** — sc_cfar ≤ τ — does not clear the conformal threshold. NOT a detection.
- **`NO_COVERAGE`** — partial-swath fingerprint (sc_ratio = 1.0 exactly with cv_ctrl = 0) or missing inference. Excluded as a missing observation rather than a non-detection.
- **`NOT_OK`** — record status flagged as an error (download failure, no_geo_meta, etc.).

## Per-site outcome counts

| Site | DETECTION | CFAR_SUPPRESSED | SUB_TAU | NO_COVERAGE | NOT_OK | Total |
|---|---|---|---|---|---|---|
| belchatow | 30 | 1 | 31 | 94 | 1 | 157 |
| boxberg | 0 | 0 | 7 | 0 | 1 | 8 |
| groningen | 1 | 0 | 5 | 0 | 0 | 6 |
| lippendorf | 1 | 0 | 7 | 0 | 0 | 8 |
| maasvlakte | 0 | 0 | 7 | 0 | 0 | 7 |
| neurath | 2 | 0 | 5 | 0 | 1 | 8 |
| rybnik | 0 | 0 | 5 | 0 | 0 | 5 |
| weisweiler | 1 | 0 | 5 | 0 | 5 | 11 |

## Detection-grade records (the only records that count as 'detections')

| Site | Date | Tile | sc_cfar | cv_ctrl | cfar_margin | source |
|---|---|---|---|---|---|---|
| weisweiler | 2021-06-01 | T31UGS | 9.813 | 1.4365 | 4.3536 | historical_backfill |
| belchatow | 2020-06-01 | T34UCB | 600.5905 | 0.7115 | 597.3061 | historical_backfill |
| belchatow | 2021-06-06 | T34UCB | 11.2571 | 1.003 | 7.0982 | historical_backfill |
| belchatow | 2024-07-10 | T34UCB | 50.6924 | 1.1052 | 46.2269 | historical_backfill |
| lippendorf | 2024-09-22 | T33UUS | 16.6898 | 0.5881 | 13.7754 | historical_backfill |
| neurath | 2024-06-25 | T32ULB | 15.26 | 0.9924 | 11.1328 | historical_backfill |
| neurath | 2024-08-29 | T32ULB | 96.9771 | 0.2879 | 94.9635 | historical_backfill |
| groningen | 2024-08-17 | T31UGV | 10.2369 | 0.6746 | 7.0632 | historical_backfill |
| belchatow | 2024-04 | T34UCB | 10.4231 | 0.6774 | 7.241 | belchatow_annual_timeseries |
| belchatow | 2024-05 | T34UCB | 126.8366 | 0.9105 | 122.9551 | belchatow_annual_timeseries |
| belchatow | 2024-07 | T34UCB | 224.4077 | 1.1465 | 219.8182 | belchatow_annual_timeseries |
| belchatow | 2024-10 | T34UCB | 5.3435 | 0.4912 | 2.72 | belchatow_annual_timeseries |
| belchatow | 2021-04 | T34UCB | 8.4943 | 0.8142 | 4.9018 | belchatow_annual_timeseries |
| belchatow | 2021-04 | T34UCB | 9.0006 | 0.6847 | 5.7963 | belchatow_annual_timeseries |
| belchatow | 2021-06 | T34UCB | 57.3596 | 0.643 | 54.2805 | belchatow_annual_timeseries |
| belchatow | 2021-07 | T34UCB | 25.56 | 0.7484 | 22.1647 | belchatow_annual_timeseries |
| belchatow | 2021-09 | T34UCB | 23.102 | 0.7451 | 19.7168 | belchatow_annual_timeseries |
| belchatow | 2021-10 | T34UCB | 3.6287 | 0.506 | 0.9608 | belchatow_annual_timeseries |
| belchatow | 2022-06 | T34UCB | 169.399 | 0.8976 | 165.5561 | belchatow_annual_timeseries |
| belchatow | 2022-07 | T34UCB | 33.167 | 1.2057 | 28.3999 | belchatow_annual_timeseries |
| belchatow | 2022-08 | T34UCB | 515.0106 | 1.2959 | 509.9729 | belchatow_annual_timeseries |
| belchatow | 2023-04 | T34UCB | 7.2525 | 0.7874 | 3.7404 | belchatow_annual_timeseries |
| belchatow | 2023-05 | T34UCB | 208.0514 | 0.3492 | 205.8538 | belchatow_annual_timeseries |
| belchatow | 2023-05 | T34UCB | 123.6368 | 0.6009 | 120.6842 | belchatow_annual_timeseries |
| belchatow | 2023-06 | T34UCB | 122.1607 | 0.5392 | 119.3932 | belchatow_annual_timeseries |
| belchatow | 2023-08 | T34UCB | 162.2144 | 1.2522 | 157.3077 | belchatow_annual_timeseries |
| belchatow | 2023-09 | T34UCB | 22.3447 | 1.2591 | 17.4174 | belchatow_annual_timeseries |
| belchatow | 2024-04 | T34UCB | 10.4231 | 0.6774 | 7.241 | belchatow_annual_timeseries |
| belchatow | 2024-05 | T34UCB | 126.8366 | 0.9105 | 122.9551 | belchatow_annual_timeseries |
| belchatow | 2024-05 | T34UCB | 150.9549 | 0.6632 | 147.8152 | belchatow_annual_timeseries |
| belchatow | 2024-07 | T34UCB | 224.4077 | 1.1465 | 219.8182 | belchatow_annual_timeseries |
| belchatow | 2024-08 | T34UCB | 91.4462 | 1.1908 | 86.7238 | belchatow_annual_timeseries |
| belchatow | 2024-09 | T34UCB | 57.0429 | 1.2965 | 52.0033 | belchatow_annual_timeseries |
| belchatow | 2024-10 | T34UCB | 5.3435 | 0.4912 | 2.72 | belchatow_annual_timeseries |
| belchatow | 2024-10 | T34UCB | 110.7504 | 1.2194 | 105.9422 | belchatow_annual_timeseries |

## CFAR-suppressed records (above τ but failing CFAR — NOT detections)

These records would be falsely flagged as detections under a τ-only rule. The production rule's CFAR gate correctly rejects them.

| Site | Date | Tile | sc_cfar | cv_ctrl | cfar_margin | source |
|---|---|---|---|---|---|---|
| belchatow | 2022-06 | T34UCB | 4.6372 | 1.4068 | -0.7332 | belchatow_annual_timeseries |
