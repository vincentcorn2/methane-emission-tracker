# Synthetic-only retraining validation

**Question:** is CH4Net v8's performance driven by the 51 synthetic plume injections, or by the 14 real positive crops? If a model trained on synthetic data alone still detects the 14 real plumes, the synthetic distribution is generalising — which validates the augmentation strategy.

## Per-crop comparison

| Crop | TROPOMI ΔXCH₄ (ppb) | v8 prob mean | v8 prob max | synth-only prob mean | synth-only prob max | v8 detect | synth-only detect |
|---|---|---|---|---|---|---|---|
| de_bad_lauchstaedt_T32UQC_20230109_enh12.npy | None | 0.584 | 0.805 | 0.000 | 0.000 | ✓ | · |
| de_bad_lauchstaedt_T32UQC_20230208_enh11.npy | None | 0.584 | 0.805 | 0.000 | 0.000 | ✓ | · |
| de_bad_lauchstaedt_T32UQC_20230216_enh16.npy | None | 0.584 | 0.805 | 0.000 | 0.000 | ✓ | · |
| de_weisweiler_T31UGS_20240918_enh18.npy | None | 0.055 | 0.481 | 0.000 | 0.000 | ✓ | · |
| nl_bergermeer_T31UFU_20230126_enh16.npy | None | 0.584 | 0.805 | 0.000 | 0.000 | ✓ | · |
| nl_grijpskerk_T31UGV_20230208_enh13.npy | None | 0.389 | 0.863 | 0.000 | 0.000 | ✓ | · |
| ro_totea_T35TLK_20230101_enh24.npy | None | 0.766 | 0.944 | 0.000 | 0.000 | ✓ | · |
| ro_totea_T35TLK_20230115_enh10.npy | None | 0.766 | 0.944 | 0.000 | 0.000 | ✓ | · |
| ro_totea_T35TLK_20230221_enh18.npy | None | 0.766 | 0.944 | 0.000 | 0.000 | ✓ | · |
| silesia_knurow_T34UCA_20230215_enh11.npy | None | 0.584 | 0.805 | 0.000 | 0.000 | ✓ | · |
| silesia_pniowek_T34UCB_20230301_enh17.npy | None | 0.318 | 0.705 | 0.000 | 0.000 | ✓ | · |
| silesia_rybnik_T34UCA_20240628_enh19.npy | None | 0.584 | 0.805 | 0.000 | 0.000 | ✓ | · |
| silesia_zofiowka_T34UCB_20230101_enh10.npy | None | 0.509 | 0.926 | 0.000 | 0.000 | ✓ | · |
| silesia_zofiowka_T34UCB_20230301_enh16.npy | None | 0.465 | 0.881 | 0.000 | 0.000 | ✓ | · |

## Summary statistics

- v8 detections on real positives: **14/14** (100%)
- Synthetic-only detections on real positives: **0/14** (0%)
- Both models agree (both detect): **0/14** (0%)

## Interpretation

**Synthetic plume training does not generalise on its own** (0% detection rate). The model trained without real positives fails to detect most real plumes, indicating the synthetic distribution is too narrow. This is a limitation to report explicitly and the synthetic generation procedure should be redesigned before deployment.
