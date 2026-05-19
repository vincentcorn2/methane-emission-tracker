# Bełchatów per-detection wind alignment

For each production-rule detection at Bełchatów, this checks whether the probability-weighted centroid of the plume sits in the direction the contemporaneous ERA5 wind would carry a plume from the mine source pin.

**N detections analysed:** 25
**N wind-aligned (angular diff < 45°):** 6 (24.0%)
**Median angular difference:** 125.9°
**Mean angular difference:** 106.0°

If the model were firing on a static terrain feature, the centroid bearing would be roughly constant across acquisitions regardless of wind direction; the expected angular_diff_deg distribution would be uniform on [0, 180] with mean 90°. A mean substantially below 90° indicates the centroid systematically tracks the wind vector, which is the expected behaviour of a real downwind plume from the mine source.

## Per-detection table

| Date | sc_cfar | Wind from (°) | Centroid bearing (°) | Δ (°) | Aligned | Dist (m) |
|---|---|---|---|---|---|---|
| 2021-04 | 8.5 | 134.0 | 56.7 | 102.7 | · | 2694 |
| 2021-04 | 9.0 | 317.5 | 90.4 | 47.1 | · | 2448 |
| 2021-06 | 57.4 | 131.1 | 100.6 | 149.5 | · | 4297 |
| 2021-07 | 25.6 | 53.6 | 93.4 | 140.2 | · | 3757 |
| 2021-09 | 23.1 | 193.5 | 99.7 | 86.2 | · | 3790 |
| 2022-07 | 33.2 | 155.8 | 94.2 | 118.4 | · | 3110 |
| 2022-08 | 515.0 | 73.6 | 69.9 | 176.3 | · | 3074 |
| 2023-04 | 7.3 | 128.2 | 97.5 | 149.3 | · | 3797 |
| 2023-05 | 208.1 | 39.4 | 93.5 | 125.9 | · | 3723 |
| 2023-05 | 123.6 | 110.6 | 92.9 | 162.3 | · | 3114 |
| 2023-06 | 122.2 | 63.6 | 95.5 | 148.1 | · | 3849 |
| 2023-08 | 162.2 | 191.6 | 96.5 | 84.9 | · | 3859 |
| 2023-09 | 22.3 | 205.8 | 100.0 | 74.2 | · | 3854 |
| 2024-04 | 10.4 | 263.4 | 45.9 | 37.5 | ✓ | 2171 |
| 2024-04 | 10.4 | 263.4 | 45.9 | 37.5 | ✓ | 2171 |
| 2024-05 | 126.8 | 99.5 | 105.7 | 173.8 | · | 1841 |
| 2024-05 | 126.8 | 99.5 | 105.7 | 173.8 | · | 1841 |
| 2024-05 | 151.0 | 113.2 | 87.4 | 154.2 | · | 3154 |
| 2024-07 | 224.4 | 295.4 | 105.4 | 10.0 | ✓ | 2347 |
| 2024-07 | 224.4 | 295.4 | 105.4 | 10.0 | ✓ | 2347 |
| 2024-08 | 91.4 | 113.1 | 107.5 | 174.4 | · | 2447 |
| 2024-09 | 57.0 | 63.0 | 112.1 | 130.9 | · | 1728 |
| 2024-10 | 5.3 | 245.4 | 79.8 | 14.4 | ✓ | 1363 |
| 2024-10 | 5.3 | 245.4 | 79.8 | 14.4 | ✓ | 1363 |
| 2024-10 | 110.8 | 121.7 | 147.1 | 154.6 | · | 1345 |