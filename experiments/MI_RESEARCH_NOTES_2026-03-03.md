# Motor Imagery Cursor — Research/Experiment Notes (2026-03-03)

## Context
Goal: robust EEG motor-imagery cursor control with weak current performance in:
- binary command classification
- active-vs-rest detection

Dataset used today:
- `03_03_26_xavier_mental_mental_command_windows.npy`
- `03_03_26_xavier_mental_mental_command_labels.npy`
- shape: `(540, 7, 300)` @ ~300 Hz, classes = `{10:neutral, 11:left, 12:right}`

## What I tested
Script: `experiments/mi_model_bakeoff.py`

Grouped CV protocol (important): hold out one full registration block per class per fold to reduce leakage from overlapping windows.

Models:
1. Riemann covariance + tangent space + logistic regression (current baseline style)
2. CSP + shrinkage LDA
3. Filter-bank Riemann + LR (8-12, 12-16, 16-20, 20-28, 8-30)

## Results (Xavier)
### 3-class (neutral / left / right)
- **CSP+LDA**: balanced acc **0.4907** (best), macro F1 **0.4906**
- Baseline Riemann+LR: balanced acc 0.4148, macro F1 0.4124
- FilterBank-Riemann+LR: balanced acc 0.4426, macro F1 0.4414

### Binary (left vs right)
- Baseline Riemann+LR: balanced acc **0.5861** (best)
- FilterBank-Riemann+LR: 0.5667
- CSP+LDA: 0.5361

### Binary (active vs neutral)
- Baseline Riemann+LR: balanced acc **0.5833** (best)
- FilterBank-Riemann+LR: 0.5736
- CSP+LDA: 0.5319

## Interpretation
- With this session’s data, **no model is good enough yet** for production cursor control.
- 3-class improves with CSP+LDA, but binary active-vs-rest remains only ~0.58 balanced acc.
- The core bottleneck likely includes data quality/protocol and feature stationarity, not only classifier choice.

## Immediate model changes made
- Added backend switch in `MentalCommandModelConfig`:
  - `backend="csp_lda"` (now default)
  - `backend="riemann_lr"`
- Updated `make_mental_command_classifier(...)` to support both paths.

## Research-backed directions to test next
1. **Session protocol hardening**
   - Keep strategy fixed per command (e.g., left-hand kinesthetic only)
   - Increase rest/control blocks and reject noisy blocks aggressively
   - Add short artifact checks (EOG/EMG contamination guard)

2. **Temporal alignment / windowing**
   - Compare MI windows around 0.5–2.5s vs 1.0–3.0s
   - Reduce overlap leakage in model selection
   - Aggregate decisions across multiple consecutive windows before cursor actuation

3. **Feature engineering upgrades**
   - FBCSP (proper feature selection per band, not simple concat)
   - Riemannian classifiers with robust covariance estimators and geometry-aware shrinkage
   - Log-variance features in individual mu/beta sub-bands per hemisphere

4. **Calibration strategy**
   - Two-stage training:
     - stage A: left-vs-right
     - stage B: active-vs-rest gate
   - Use confidence-gated output (no move when confidence low)

5. **Subject adaptation**
   - Incremental re-fit every N accepted live windows
   - Weighted replay of recent samples to combat drift

6. **Control-level fixes**
   - Treat MI decoder as velocity intent, not direct hard class
   - Add dead-zone + hysteresis + dwell-time to reduce jitter

## Recommendation for next run
- Use `backend="csp_lda"` for 3-class experiments.
- Add a dedicated binary gate model (active vs neutral) before left/right classifier.
- Collect 2–3 more subject sessions with stricter block acceptance before selecting deep models (EEGNet etc.).
