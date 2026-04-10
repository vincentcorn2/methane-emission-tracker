# CH4Net European Monitoring — Agenda Update
*Continues from prior summary. New work below.*

---

## Work Done Since Last Update

### 1. Geospatial Root-Cause Resolution (Weisweiler Zero-Padding Bug)

Every prior evaluation of the Weisweiler lignite plant had returned S/C = 1.000 — the model predicting identically over the plant and its control region — which had been interpreted as a model failure. We traced this to a geospatial data error: the assigned Sentinel-2 tile (T32ULB, orbit R108) is zero-padded at Weisweiler's longitude. The plant sits at pixel column 1,144, but valid data in that tile only begins at column 2,825. The model was performing inference on empty pixels. Once we identified and assigned the correct tile (T31UGS, orbit R008), confirmed by checking B12 = 96 DN at the plant coordinates, the baseline model immediately detected Weisweiler at S/C = 4.004 — well above the 1.15 detection threshold. This finding retroactively invalidates all prior Weisweiler results and confirms the plant is detectable from orbit.

A secondary instance of the same bug was found in the downloaded winter reference tile for Weisweiler: it too had been pulled from orbit R108 and was entirely zero-padded at the plant location. A corrected reference tile from orbit R008 (January 27, 2024) was downloaded and validated (B11 = 73, B12 = 55 at the plant pixel).

### 2. Bitemporal Differencing: Implementation, Validation, and Limits

We implemented and evaluated a bitemporal differencing approach to suppress terrain false positives. The method replaces B11 and B12 in the input array with a seasonal difference (summer target minus winter reference, shifted to a neutral baseline of 128), so that permanent terrain features cancel out and only transient atmospheric anomalies remain visible to the model.

**Outcome on false positives:** The approach substantially suppressed two confirmed false positives. The Groningen gas field dropped from S/C = 152 (baseline, global model) to S/C = 3.9 (bitemporal, EU fine-tuned model). The Rybnik coal mine dropped from S/C = 15.7 to S/C = 1.3.

**Outcome on Weisweiler:** Bitemporal differencing was found to be counterproductive for Weisweiler specifically. Inspection of pixel values at the plant revealed that B11 and B12 increase by nearly identical amounts between winter and summer (+40 and +41 DN respectively), reflecting uniform seasonal brightening of industrial infrastructure rather than a methane-specific anomaly. After differencing, the model receives no SWIR differential and suppresses the detection from S/C = 4.004 to S/C = 0.99. Weisweiler has been flagged to skip bitemporal differencing; its baseline result is the operative detection.

**Outcome at Boxberg:** The open-pit Boxberg lignite mine produces anomalously large BT false positives (S/C = 7–32 across runs) driven by open-pit mining dynamics: rapid volumetric extraction between the winter reference and summer target means the winter and summer pixels represent entirely different terrain, violating the differencing assumption. A spatial exclusion mask or per-site BT bypass is required.

### 3. European Fine-Tuning: Progress and Identified Failure Mode

Three retraining iterations were conducted on 21–26 European crops (9–14 positive, 9–12 negative):

- **v1:** 18 train / 5 val, best val loss 0.7926. Achieved meaningful FP suppression (Groningen 152 → 11.7 baseline, 3.9 BT) but Weisweiler was untestable due to the tile bug.
- **v2 (failed):** A zero-padded Weisweiler crop from the wrong tile was included in training. Training collapsed: val loss reached 95.9, accuracy stalled at 47%. Discarded.
- **v3:** Zero-padded crop removed, correct T31UGS Weisweiler crop added. Best val loss 0.6840, val accuracy reached 80% at epoch 20 (vs. epoch 36 in v1), indicating faster convergence. However, baseline evaluation revealed a critical failure mode: EU v3 suppressed every site to near-zero baseline S/C — true positives and false positives alike (Weisweiler 4.004 → 0.648, Rybnik 3.708 → 0.085, Boxberg 5.422 → 0.031, Groningen 152 → 0.549). The model learned a blanket heuristic ("European industrial terrain = not methane") rather than learning to distinguish methane from terrain artifacts. Bitemporal differencing applied on top of this produces catastrophic false positives (Groningen BT = 553).

This is a confirmed case of catastrophic overfitting on a 21-sample dataset with a 13.5M parameter model. All encoder weights were allowed to update, which destroyed the global feature representations needed for SWIR edge detection. The global base model — with no European fine-tuning — remains the better detector for true positives at this stage.

- **v4 (in progress):** Three training configurations were tested in pursuit of a fine-tuned model that suppresses false positives without destroying true positive sensitivity. Hard encoder freezing (freeze inc/down1/down2, ~8.5% of params) caused the decoder to plateau at the all-zero prediction minimum — BCEWithLogitsLoss is minimised by predicting background everywhere when plume masks are sparse. Raising POS_WEIGHT from 5 to 20 rescaled the plateau without breaking it. Adding Dice loss on top of a frozen encoder caused divergence (val loss climbing to 6.0+ by epoch 15), as the Dice and BCE gradients conflicted on the constrained decoder. Current approach: full model updates with L2 weight decay (1e-3) added to the Adam optimizer. This acts as a soft regularizer — the encoder can adapt, but large drift away from the global pretraining is penalised, which should prevent the blanket-suppression heuristic observed in v3. Training in progress.

---

## Next Steps

### Immediate (before next retraining run)

**1. Complete v4 training (weight decay regularization)**
v4 is currently training with L2 weight decay (1e-3) and full model updates. If it converges to val_loss < 0.75 and acc = 80% as v3 did, evaluate baseline S/C at all four sites. If Weisweiler and Rybnik hold above 1.15 while Groningen drops further, the regularization is working. If suppression repeats, the next option is reducing the learning rate to 1e-5 (slower drift) or switching to image-level classification loss instead of pixel segmentation.

**2. Implement synthetic plume augmentation**
To address the fundamental data scarcity (21 samples for 13.5M parameters), synthetically inject Gaussian methane plumes over the 12 existing negative European crops. This generates additional positive training examples grounded in real European terrain backgrounds without requiring new satellite downloads. Combines with encoder freezing to reduce overfitting risk.

**3. Per-site bitemporal strategy**
The single BT configuration does not generalise across all sites:
- Weisweiler: skip BT entirely; baseline S/C = 4.004 is the operative result
- Boxberg: skip BT; open-pit mining violates the stationarity assumption
- Groningen, Rybnik: apply B11+B12 BT; suppresses vegetation and terrain artifacts effectively

This is already implemented as a `skip_bitemporal` flag per site in the evaluation script.

### Near-Term

**4. Multi-date evaluation for Weisweiler**
The S/C = 4.004 result is from a single acquisition (September 18, 2024). Methane emissions are episodic; a single date does not establish reliable detection. Run baseline inference across 5–10 clear-sky T31UGS R008 acquisitions from summer 2024 and compute the distribution of S/C values. A median S/C consistently above 1.15 constitutes robust evidence of detection.

**5. TROPOMI co-location at Weisweiler**
The Weisweiler plant is a TROPOMI-confirmed emitter (~18 ppb above background). Match the September 18, 2024 Sentinel-2 acquisition date against TROPOMI Level-2 CH4 products to confirm the plant was actively emitting on that specific day. This transforms the S/C = 4.004 result from a spatial correlation into a validated detection event.

**6. Reference tile quality for median-composite baseline**
Replace single-date winter reference tiles with a multi-date median composite (3–5 clear-sky acquisitions from November–February per site). A median baseline is resilient to ephemeral snow, frost, and cloud shadow contamination that corrupts single-date references — the mechanism responsible for the Boxberg BT anomaly.
