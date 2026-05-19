# CH4Net v8 — standard ML segmentation metrics

Production rule on the inference output: probability ≥ 0.18, contiguous region ≥ 115 px.


## In-sample evaluation (training crops)

- Positive crops: 65
- Negative crops: 22

### Pixel-level metrics on positive crops
- Mean IoU: **0.257**
- Mean precision: **0.284**
- Mean recall: **0.936**
- Mean F1: **0.392**

### Scene-level (does the crop fire?)
- Positive recall: **100.0%** (65/65)
- Negative FPR: **63.6%** (14/22)

### Discriminative power
- AUROC: **0.862**
- Average precision (PR-AUC): **0.900**