# Second Take: MI/3-Class Experiments + Recommendations (2026-03-04)

## Why this second take

After reviewing all local experiment outputs and comparing against known BCI literature (CSP/FBCSP, EEGNet, cross-dataset variability), the core conclusion is:

- The current pipeline is **technically reasonable**, but the biggest bottlenecks are **data protocol + evaluation design**, not a missing fancy model.

---

## 1) Re-interpretation of current results

## Binary left/right (live/MI/handclench `.npy`)

Best aggregate model across available datasets remains:
- **Riemannian Tangent Space + Logistic Regression (`C=0.1`)**
- Mean balanced accuracy ~**0.56**

This is consistent with expected cross-session/cross-subject MI difficulty.

## 3-class (neutral/left/right)

Two different numbers exist because of split strategy:

- **Optimistic non-grouped window CV** (Xavier mental windows):
  - 3-class ≈ **0.745 bal acc**
  - L/R only ≈ **0.818 bal acc**

- **More realistic grouped-by-block CV**:
  - Xavier 3-class ~**0.435–0.478** (best with 2-stage + smoothing)
  - Winston 3-class ~**0.398–0.429**

Interpretation: your data *does* contain L/R signal, but neutral/rest separation and leakage-sensitive evaluation are dominating outcomes.

---

## 2) Why Omer looked strong

Omer's set (`handclench_real`) likely benefits from:
1. Overt movement-like physiology (stronger discriminative rhythm changes than pure imagery)
2. Cleaner class structure in covariance space (favors Riemannian methods)
3. Balanced labels and enough trials for that task
4. Wider filter bank helping capture subject-specific informative frequencies

This does **not** automatically transfer to imagery-only left/right online control.

---

## 3) What the literature says that matches your case

- **CSP is strong but frequency/time-window sensitive** (Ramoser 2000; FBCSP work on BCI Comp IV): subject-specific band and timing matter a lot.
- **Filter-bank CSP/Riemannian often beats single-band** when tuned per subject.
- **EEGNet** is useful in low-data EEG settings, but evaluation must avoid leakage and dataset shift artifacts.
- **Cross-dataset variability is a major issue** in MI decoding; alignment/domain adaptation can help.

(References gathered via web fetch: Ramoser et al. 2000 IEEE; Ang et al. FBCSP Frontiers 2012; EEGNet arXiv 1611.08024; Frontiers 2020 cross-dataset variability.)

---

## 4) High-impact improvements to data collection

## A. Make trial structure evaluation-safe by design

Current issue: many overlapping windows per same block create leakage risk.

Recommendation:
- Save explicit metadata for each epoch/window:
  - `subject_id`, `session_id`, `block_id`, `trial_id`, `class`, `timestamp`
- In all reports, default to:
  - **group by block** for within-session
  - **group by session** for generalization
  - **leave-one-subject-out** for transfer claims

## B. Improve MI cue protocol (for cleaner physiology)

- Add fixation baseline period and jittered ITI
- Keep constant cue timing; randomize class order in balanced mini-blocks
- Record subjective compliance per block (easy/hard/focused)
- Add short breaks every 20–30 trials to prevent fatigue drift

## C. Improve channel strategy

Current sets are often sparse/non-standard (e.g., 2-channel Winston set).

For MI cursor control, prioritize channels around sensorimotor cortex:
- Minimum: `C3, Cz, C4`
- Better: `FC3/FCz/FC4, C3/Cz/C4, CP3/CPz/CP4`

If hardware limits channels, include **Cz** whenever possible.

## D. Capture resting/neutral properly

Rest class is currently weak and confounded.

Use distinct neutral protocol:
- Eyes-open fixation + explicit no-imagery instruction
- Balanced neutral trial count (not underrepresented)
- Separate neutral blocks from active blocks to avoid transition contamination

---

## 5) High-impact improvements to training/evaluation

## A. Standardized evaluation ladder (must-have)

For each model, always report:
1. **Window-level random CV** (diagnostic only)
2. **Grouped-by-block CV** (primary)
3. **Session holdout** (realistic)
4. **Subject holdout** (hard transfer)

Only use (2)-(4) for decision-making.

## B. Move to hierarchical runtime for 3-class control

Instead of flat 3-way output:
- Stage 1: `active vs neutral`
- Stage 2: `left vs right` (only when active)
- Add short causal smoothing (3–5 windows)

This already improved grouped results in your runs.

## C. Per-subject hyperparameter search (small, strict)

Tune these with grouped CV only:
- frequency bands (filter-bank grid)
- epoch window (`tmin/tmax`)
- regularization strength (`C`)
- confidence thresholds

Avoid large unguided sweeps; keep reproducible search spaces.

## D. Add alignment/domain adaptation for transfer

Given cross-subject variability:
- Riemannian re-centering / covariance alignment
- Session-wise z-scoring / robust scaling
- Fine-tune with small subject-specific calibration batches

## E. Online retraining policy

Only retrain using high-quality samples:
- artifact-free
- high-confidence predictions
- balanced recent buffer per class

Do not blindly append all online samples.

---

## 6) Suggested roadmap (practical)

### Phase 1 (this week): reliability first
- Enforce grouped evaluation in all scripts
- Add metadata + block/session ids to saved datasets
- Build one unified benchmark script with fixed seeds/reports

### Phase 2: data quality
- Improve MI protocol and channel placement
- Collect 2–3 clean sessions per subject with standardized format
- Rebuild train/test splits using session-level holdout

### Phase 3: model upgrades
- Compare:
  - Riemannian LR baseline
  - FBCSP + LDA
  - EEGNet (with strict grouped/session split)
- Keep best by grouped/session metrics, not random-window CV

---

## 7) Bottom line

- You already have a workable baseline.
- Biggest gains will come from **better protocol + leakage-safe evaluation + neutral-class design**, not from swapping classifiers alone.
- For immediate deployment, keep:
  - binary L/R model for control,
  - optional hierarchical neutral gate,
  - strict grouped/session validation as the acceptance criterion.
