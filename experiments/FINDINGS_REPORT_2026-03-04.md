# ErrP/MI Modeling Findings Report (2026-03-04)

## Executive Summary

I ran two tracks of experiments:

1. **Binary left/right MI** on `*_data.npy` / `*_labels.npy` datasets from the Drive BCI folder.
2. **3-class (neutral/left/right)** on mental-command windows datasets.

### Key outcomes

- **Binary across available MI datasets is still modest** (aggregate balanced accuracy ~0.56 best model).
- **Best overall binary model**: Riemannian Tangent Space + Logistic Regression (`C=0.1`).
- **Omer dataset was strongest** (up to ~0.64 bal acc), likely due to stronger overt motor signal.
- **3-class Xavier “high” score (0.745 bal acc)** came from a non-grouped repeated CV setup and is likely optimistic due to window-level leakage.
- **More realistic grouped-by-block 3-class results** are lower (~0.435–0.478 on Xavier depending on postprocessing).

---

## Data used

### Binary MI/hand data (from `data/drive_dump/BCI`)

- `03_03_26_winston_live_data.npy` (120, 2, 900), labels {1,2}
- `03_03_26_xavier_live_data.npy` (110, 7, 600), labels {1,2}
- `matthew_25_02_26_mi_data.npy` (120, 7, 600), labels {1,2}
- `omer_2_23_26_handclench_real_data.npy` (70, 7, 600), labels {1,2}
- `yichen_2_25_26_handclench_real_data.npy` (80, 7, 600), labels {1,2}

### 3-class mental command

- `03_03_26_xavier_mental_mental_command_windows.npy` (540, 7, 300), labels {10,11,12}
- `03_03_26_winston_mental_mental_command_windows.npy` (419, 2, 300), labels {10,11,12}

---

## Binary MI benchmark results

Source: `experiments/results/mi_best_from_drive.json`

Models tested:
- `riemann_lr`
- `riemann_lr_C0.1`
- `csp4_lda`
- `csp6_lda`
- `fbank_riemann`
- `fbank_riemann_wide`

### Aggregate mean balanced accuracy (across 5 datasets)

1. `riemann_lr_C0.1`: **0.5637**
2. `riemann_lr`: 0.5528
3. `csp6_lda`: 0.5487
4. `fbank_riemann_wide`: 0.5415
5. `fbank_riemann`: 0.5402
6. `csp4_lda`: 0.5238

### Best per dataset

- Winston live: `fbank_riemann` = **0.525**
- Xavier live: `fbank_riemann_wide` = **0.520**
- Matthew MI: `riemann_lr_C0.1` = **0.515**
- Omer handclench real: `fbank_riemann_wide` = **0.640**
- Yichen handclench real: `riemann_lr_C0.1` = **0.6525**

---

## Why Omer did better

Likely reasons for Omer’s stronger performance:

1. **Task type**: `handclench_real` (overt movement) is typically easier to decode than pure imagined movement.
2. **Signal separability**: Covariance structure appears more separable for his classes, which benefits Riemannian methods.
3. **Class balance is clean**: 35/35 labels (in npy set), reducing class imbalance confounds.
4. **Filter-bank gain**: Wider bands (`6–10, 10–14, 14–18, 18–24, 24–32, 8–30`) helped capture discriminative low+mu/beta content.

In short: overt motor behavior + covariance-based features = better class separation.

---

## 3-class optimization results

### A) Optimistic (non-grouped repeated stratified CV)

Source: `experiments/results/mi_3class_optimize.json`

For Xavier mental (7ch):
- Flat 3-class filter-bank+LR: **0.745 bal acc**, F1 0.746
- Hierarchical 2-stage: 0.720 bal acc, F1 0.722
- Binary L/R only (no neutral): **0.818 bal acc**

### B) More realistic grouped-by-block OOF results

Source:
- `experiments/results/xavier_mental_3class_robust.txt`
- `experiments/results/winston_mental_3class_robust.txt`

Xavier grouped OOF:
- Single-stage riemann: 0.4148
- Single-stage + robust preproc: 0.4352
- Two-stage best: 0.4556
- Two-stage + smoothing (window=3/5): **0.4778**

Winston grouped OOF:
- Single-stage riemann: 0.3959
- Two-stage + smoothing (window=5): **0.4286**

---

## Important caveat on “high Xavier accuracy”

The **0.745** value is from repeated stratified splitting at the **window level**, which can mix windows from the same registration blocks into both train/test.
That setup can overestimate real-world performance.

For deployment realism, grouped-by-block/session evaluations are more trustworthy, and those are currently in the **0.43–0.48** range for 3-class.

---

## Reaching data status

- The Reaching folder appears to contain mostly raw `.csv/.edf/.dsi` files.
- I added `experiments/parse_raw_to_epochs.py` to convert trigger-marked raw CSV into epoched `*_data.npy` and `*_labels.npy`.
- Need additional Reaching raw files downloaded successfully (Drive rate/permission limits interrupted some fetches) to integrate into training.

---

## Scripts and artifacts added

- `experiments/mi_best_from_drive.py`
- `experiments/results/mi_best_from_drive.json`
- `experiments/mi_3class_optimize.py`
- `experiments/parse_raw_to_epochs.py`
- `experiments/results/mi_3class_optimize.json`
- `experiments/results/xavier_mental_3class_robust.txt`
- `experiments/results/winston_mental_3class_robust.txt`

---

## Recommended next steps

1. **Use grouped CV as primary metric** (block/session holdout).
2. Add more 7-channel MI sessions (especially reaching left/right) converted to npy.
3. Keep two-stage decoding for 3-class runtime:
   - Stage 1: active vs neutral
   - Stage 2: left vs right
   - causal smoothing (3–5 window)
4. Evaluate per-subject calibration + transfer (LOSO).
5. Expand channel set around C3/Cz/C4 if hardware allows to boost MI separability.
