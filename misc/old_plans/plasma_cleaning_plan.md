# Plasma Data Cleaning Plan

Source: `data/plasma_recordings.xlsx`
Paper: *Neural correlates of the DMT experience assessed with multivariate EEG* (Timmermann et al., 2019)

---

## Source Data Structure

The Excel file (`Sheet1`) contains two parallel side-by-side tables — DMT and Placebo — each split across two row blocks:

| Block | Rows | Content |
|---|---|---|
| Concentrations | 5–17 | Plasma DMT/placebo values (nmol/L) per subject |
| Dose groups | 5–17, col A | Dose label (7 mg / 14 mg / 18 mg / 20 mg) — only on first row of each group |
| Averages | 18, 20–22 | AVG / Low / Medium / High group means (used to impute outliers) |
| Time points | 26–38 | Blood draw times in minutes per subject (5 draws each) |
| Time averages | 39, 41–43 | Averages of time points per dose group |

**Columns:**
- DMT: cols B–G → `[subject, t0, t1, t2, t3, t4]` (concentrations), col A → dose label
- Placebo: cols J–O → `[subject, t0, t1, t2, t3, t4]` (concentrations)
- Times follow the same column layout in rows 26–38

**Subjects:** 13 total (S01–S13, no S... note: S08 listed after S06 in 14 mg group)

**Dose groups:**
| Dose | Subjects |
|---|---|
| 7 mg | S01, S02, S03 |
| 14 mg | S04, S05, S06, S08 |
| 18 mg | S11 |
| 20 mg | S07, S09, S10, S12, S13 |

**Imputed cells:** Outliers and NaNs were replaced with group-mean formulas (e.g. `=C18`, `=E20`). These resolve correctly when loading with `data_only=True`.

---

## Known Data Issues

| Issue | Detail |
|---|---|
| Formula cells | ~12 cells contain `=AVERAGE(...)` references to group means instead of raw values |
| Dose column sparse | Dose label only appears on first subject of each group — needs forward-fill |
| Irregular time points | Each subject's 5 blood draws happen at different minutes |
| Wide format | Concentrations and times are in separate row blocks, not paired |
| S13 DMT baseline = 0 | Likely a true zero (pre-injection), not missing |
| Floating-point imputation | Some imputed values have floating-point noise (e.g. `123.76666...`) |

---

## Target Format

A single **tidy long-format** DataFrame — one row per measurement:

```
subject | condition | dose_mg | time_point | time_min | plasma_conc | is_imputed
  S01   |    dmt    |    7    |     0      |    0.0   |    0.207    |   False
  S01   |    dmt    |    7    |     1      |    3.0   |   94.100    |   False
  S01   |    dmt    |    7    |     2      |    5.0   |   50.900    |   False
  S01   |    dmt    |    7    |     3      |    9.0   |   18.400    |   False
  S01   |    dmt    |    7    |     4      |   20.0   |    5.890    |   False
  S03   |    dmt    |    7    |     2      |    5.0   |   59.550    |   True   ← imputed
  S01   | placebo   |   NaN  |     0      |    0.0   |    1.310    |   False
  ...
```

**Column types:**
- `subject`: string (`"S01"` … `"S13"`)
- `condition`: string (`"dmt"` or `"placebo"`)
- `dose_mg`: Int64 nullable integer (NaN for placebo)
- `time_point`: int (0–4, index of the blood draw)
- `time_min`: float (actual minutes since injection)
- `plasma_conc`: float (nmol/L)
- `is_imputed`: bool

---

## Cleaning Steps

### Step 1 — Dual load
Load the workbook **twice**:
- `data_only=True` → resolved numeric values (use these as the actual data)
- `data_only=False` → detect formula cells → mark as `is_imputed=True`

### Step 2 — Parse DMT concentrations (rows 5–17, cols A–G)
- Forward-fill col A (dose label) down through each group
- Extract `[subject, dose_mg, c0, c1, c2, c3, c4]`
- Strip the ` mg` suffix and cast dose to int

### Step 3 — Parse DMT time points (rows 26–38, cols B–G)
- Extract `[subject, t0, t1, t2, t3, t4]`
- Match to concentrations by subject ID

### Step 4 — Melt DMT to long format
- Stack the 5 (time_point, concentration) pairs per subject into rows
- Result: one row per draw

### Step 5 — Repeat Steps 2–4 for Placebo (cols J–O, same row ranges)
- `dose_mg` = `pd.NA`

### Step 6 — Flag imputed values
- Cross-reference formula-cell positions from Step 1
- Set `is_imputed=True` for matching `(subject, condition, time_point)` combinations

### Step 7 — Concatenate & validate
- `pd.concat([dmt_long, placebo_long])`
- Assert: 13 subjects × 5 draws × 2 conditions = 130 rows total
- Assert: no NaN in `plasma_conc` or `time_min`
- Assert: all `time_min` values ≥ 0
- Round imputed floating-point values to 4 decimal places

### Step 8 — Export
```
data/plasma_clean.csv      ← human-readable, version-controllable
data/plasma_clean.parquet  ← efficient for pandas/polars pipelines
```

---

## Output Files

| File | Purpose |
|---|---|
| `data/plasma_cleaning_plan.md` | This document |
| `data/clean_plasma.py` | Reproducible cleaning script |
| `data/plasma_clean.csv` | Tidy long-format output |
| `data/plasma_clean.parquet` | Binary format for fast loading |

---

## Usage After Cleaning

```python
import pandas as pd

df = pd.read_parquet("data/plasma_clean.parquet")

# Per-subject DMT curve (ready for interpolation)
dmt = df[df["condition"] == "dmt"]
subject = dmt[dmt["subject"] == "S05"].sort_values("time_min")
times = subject["time_min"].values        # x for interpolation
concs = subject["plasma_conc"].values     # y for interpolation
```
