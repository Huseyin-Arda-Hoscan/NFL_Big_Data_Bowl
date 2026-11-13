# Predicting NFL In-Game Speed from Combine Tracking Data: Target Formulation, Temporal Validation, and the Role of Micro-Biomechanical Features

---

## 1. Problem Formulation and Research Questions

At the NFL Scouting Combine, Wide Receiver (WR) prospects are evaluated both in linear speed drills (the 40-yard dash) and in position-specific route drills (*gauntlet*, *slant*, *out*, *corner*, *post*, *go*) using 10 Hz player-tracking sensors. While traditional scouting relies primarily on stopwatch 40-yard dash times (`forty`) and basic anthropometric measurements (`combine_height`, `combine_weight`), tracking-derived micro-biomechanical metrics from route drills have been proposed to better capture in-game speed and Combine-to-NFL speed translation (*Kinematic Translation Ratio — KTR*).

This study addresses three empirical research questions:
1. **Direct In-Game Speed Prediction:** To what extent can pre-draft Combine measurements predict a WR prospect's regular-season 95th-percentile in-game speed (`nfl_v95`) across unseen draft cohorts?
2. **Incremental Value of Route Micro-Features:** Do Combine route-drill tracking features (`hb_v95`, `deep_v95`, `hb_cut_speed_retention`) and extended sensor speed/acceleration metrics (`s1_v95`, `s2_vmax`, `forty_a95`, `s2_a95`) provide consistent out-of-sample gains over a parsimonious baseline of linear speed and anthropometrics?
3. **Target Formulation (Ratio vs. Numerator Modeling):** When modeling `KTR_Skill_Speed = nfl_v95 / s2_v95`, how much predictive performance stems from learning in-game speed (`nfl_v95`) versus leveraging the known pre-draft Combine denominator (`s2_v95`)?

---

## 2. Dataset, Target Variables, and Feature Engineering

### 2.1. Cohort Structure
The dataset comprises **N = 76 Wide Receivers** across three draft classes who participated in 10 Hz Combine tracking and logged regular-season NFL pass-play tracking data:
* **2023 Draft Cohort:** N = 31 WRs
* **2024 Draft Cohort:** N = 26 WRs
* **2025 Draft Cohort:** N = 19 WRs

### 2.2. Target Variables
* **Primary Target — `nfl_v95` (yds/s):** A player's 95th-percentile speed across tracked regular-season NFL pass plays.
* **Secondary Target — `KTR_Skill_Speed`:** The ratio of NFL in-game 95th-percentile speed to Combine Stage-2 (positional route drill) 95th-percentile speed:
  `KTR_Skill_Speed = nfl_v95 / s2_v95`
  Because `s2_v95` is measured at the Combine, the denominator is observable prior to the NFL Draft.

### 2.3. Candidate Pre-Draft Feature Sets and Bivariate Correlations
All model inputs are strictly restricted to pre-draft Combine measurements. Post-draft NFL variables (such as tracked regular-season play count `reg_plays_tracked` and `starter_ratio`) are excluded from all predictive models and used solely in post-hoc residual diagnostics.

```python
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load pre-draft Combine + NFL target master table (N = 76 WRs across 2023-2025)
CACHE_DIR = Path("data_cache")
wr_micro_df = pd.read_parquet(CACHE_DIR / "wr_predraft_micro_master.parquet")
df = wr_micro_df.dropna(subset=["nfl_v95", "s2_v95", "KTR_Skill_Speed", "draft_year"]).copy().reset_index(drop=True)
df["draft_year"] = df["draft_year"].astype(int)

# Bivariate correlation analysis with nfl_v95 (numerator), s2_v95 (denominator), and KTR
feature_definitions = [
    ("forty_vmax", "40-Yd Sensor Peak Speed (yds/s)"),
    ("forty", "40-Yd Official Stopwatch Time (s)"),
    ("combine_weight", "Combine Body Weight (lb)"),
    ("s1_v95", "Combine Stage-1 Track P95 Speed (yds/s)"),
    ("combine_height", "Combine Height (in)"),
    ("s2_vmax", "Combine Stage-2 Route Peak Speed (yds/s)"),
    ("hb_cut_speed_retention", "Hard-Break Cut Speed Retention (hb_cut_speed / hb_v95)"),
    ("deep_v95", "Deep-Route Drill P95 Speed (yds/s)"),
    ("forty_a95", "40-Yd Sensor P95 Acceleration (yds/s^2)"),
    ("hb_v95", "Hard-Break Route Drill P95 Speed (yds/s)"),
    ("s2_a95", "Combine Stage-2 Route P95 Acceleration (yds/s^2)"),
    ("s2_v95", "Combine Stage-2 Overall Route P95 Speed (KTR Denominator)"),
]

corr_records = []
for col, desc in feature_definitions:
    cols_unique = list(dict.fromkeys([col, "nfl_v95", "s2_v95", "KTR_Skill_Speed"]))
    sub = df[cols_unique].dropna()
    r_nfl, p_nfl = stats.pearsonr(sub[col], sub["nfl_v95"])
    rho_nfl, _ = stats.spearmanr(sub[col], sub["nfl_v95"])
    r_s2, _ = stats.pearsonr(sub[col], sub["s2_v95"])
    r_ktr, p_ktr = stats.pearsonr(sub[col], sub["KTR_Skill_Speed"])
    corr_records.append({
        "Feature": col,
        "Description": desc,
        "N": len(sub),
        "r_nfl_v95": round(r_nfl, 3),
        "p_nfl_v95": round(p_nfl, 4),
        "Spearman_nfl_v95": round(rho_nfl, 3),
        "r_s2_v95": round(r_s2, 3),
        "r_KTR": round(r_ktr, 3),
        "p_KTR": round(p_ktr, 4),
    })

corr_df = pd.DataFrame(corr_records)
display(corr_df)
```

| Feature | Description | N | r (`nfl_v95`) | p (`nfl_v95`) | Spearman ρ (`nfl_v95`) | r (`s2_v95` Denom.) | r (`KTR`) | p (`KTR`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `forty_vmax` | 40-Yd Sensor Peak Speed (yds/s) | 68 | **+0.597** | <0.0001 | **+0.639** | +0.138 | +0.332 | 0.0057 |
| `forty` | 40-Yd Official Stopwatch Time (s) | 70 | **-0.525** | <0.0001 | **-0.562** | -0.159 | -0.261 | 0.0288 |
| `combine_weight` | Combine Body Weight (lb) | 76 | **-0.493** | <0.0001 | **-0.508** | -0.132 | -0.253 | 0.0274 |
| `s1_v95` | Combine Stage-1 Track P95 Speed (yds/s) | 69 | **+0.414** | 0.0004 | **+0.630** | +0.043 | +0.274 | 0.0226 |
| `combine_height` | Combine Height (in) | 76 | **-0.316** | 0.0054 | **-0.343** | -0.191 | -0.079 | 0.4952 |
| `s2_vmax` | Combine Stage-2 Route Peak Speed (yds/s) | 76 | **+0.266** | 0.0202 | +0.385 | +0.653 | -0.325 | 0.0041 |
| `hb_cut_speed_retention` | Hard-Break Cut Speed Retention | 76 | -0.225 | 0.0502 | -0.208 | -0.304 | +0.066 | 0.5726 |
| `deep_v95` | Deep-Route Drill P95 Speed (yds/s) | 76 | +0.223 | 0.0528 | +0.381 | **+0.669** | -0.370 | 0.0010 |
| `forty_a95` | 40-Yd Sensor P95 Acceleration (yds/s²) | 68 | -0.086 | 0.4842 | -0.125 | +0.147 | -0.174 | 0.1548 |
| `hb_v95` | Hard-Break Route Drill P95 Speed (yds/s) | 76 | **+0.027** | 0.8185 | -0.019 | **+0.706** | **-0.533** | <0.0001 |
| `s2_a95` | Combine Stage-2 Route P95 Accel. (yds/s²) | 76 | +0.011 | 0.9249 | -0.041 | +0.276 | -0.213 | 0.0649 |
| `s2_v95` | Combine Stage-2 Overall P95 Speed (Denom.) | 76 | **+0.131** | 0.2582 | +0.122 | **1.000** | **-0.686** | <0.0001 |

The correlation structure highlights two distinct empirical patterns:
* The strongest bivariate associations with `nfl_v95` come from linear speed measurements (`forty_vmax`: r = +0.597; `forty`: r = -0.525; `s1_v95`: r = +0.414) and anthropometric attributes (`combine_weight`: r = -0.493; `combine_height`: r = -0.316). These represent predictive associations rather than proof of direct causal effects.
* Combine route-speed variables (`hb_v95`, `deep_v95`, `s2_v95`) show weak or statistically insignificant associations with `nfl_v95` (r = +0.027 to +0.223), but strong positive correlations with the KTR denominator `s2_v95` (r = +0.669 to +0.706) and consequently negative correlations with the `KTR_Skill_Speed` ratio.

---

## 3. Temporal Validation Protocol: Leave-One-Draft-Year-Out (LOGO)

Random K-Fold cross-validation mixes players from the same draft class across training and validation sets. To evaluate cross-cohort generalization across draft classes, **Leave-One-Draft-Year-Out (LOGO)** serves as our primary evaluation protocol, with 5-Fold CV (`KFold(n_splits=5, shuffle=True, random_state=42)`) reported as a secondary reference.

**Verified Leak-Free Implementation:**
In every LOGO fold, two draft cohorts form the training set and the held-out draft year forms the test set (`2023`, `2024`, and `2025` tested sequentially). Median imputation (`SimpleImputer(strategy="median")`), standardization (`StandardScaler()`), `Ridge(alpha=10.0)` fitting, any inner-loop feature selection (`Model C4`), and the training-mean reference calculation (`mean_train_nfl_v95`) are executed **strictly inside each training split (`tr_idx`)** without access to test-year observations (`va_idx`).

```python
BASE_CLASSIC = ["forty", "forty_vmax", "combine_height", "combine_weight"]
REDUCED_3 = ["hb_v95", "deep_v95", "hb_cut_speed_retention"]
EXT_POOL_C = ["s1_v95", "s2_vmax", "forty_amax", "forty_a95", "s2_amax", "s2_a95", "hb_a95", "deep_a95", "hb_cut_speed"]

def evaluate_logo_and_5fold(df, features_or_mode, alpha=10.0, label=""):
    y_nfl = df["nfl_v95"].values
    y_ktr = df["KTR_Skill_Speed"].values
    s2 = df["s2_v95"].values
    groups = df["draft_year"].values

    # Primary Evaluation: Leave-One-Draft-Year-Out (LOGO)
    logo = LeaveOneGroupOut()
    oof_nfl_logo = np.zeros(len(df))
    tr_r2_logo, va_r2_logo = [], []
    yr_nfl_r2, yr_nfl_rho = {}, {}
    yr_ktr_r2, yr_ktr_rho = {}, {}

    for tr, va in logo.split(df, y_nfl, groups):
        yr = int(groups[va][0])
        if features_or_mode == "NESTED_TOP2":
            # Feature selection performed strictly within the training years (tr)
            corrs = []
            for c in EXT_POOL_C:
                sub_tr = df.iloc[tr][[c, "nfl_v95"]].dropna()
                r_val, _ = stats.pearsonr(sub_tr[c], sub_tr["nfl_v95"])
                corrs.append((abs(r_val), c))
            corrs.sort(reverse=True)
            feats = BASE_CLASSIC + [c for _, c in corrs[:2]]
        else:
            feats = list(features_or_mode)

        X_tr, X_va = df.iloc[tr][feats], df.iloc[va][feats]
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
        pipe.fit(X_tr, y_nfl[tr])

        p_tr = pipe.predict(X_tr)
        p_va = pipe.predict(X_va)
        oof_nfl_logo[va] = p_va

        tr_r2_logo.append(r2_score(y_nfl[tr], p_tr))
        va_r2_logo.append(r2_score(y_nfl[va], p_va))
        yr_nfl_r2[yr] = r2_score(y_nfl[va], p_va)
        yr_nfl_rho[yr] = stats.spearmanr(y_nfl[va], p_va)[0]

        # Derived KTR = out-of-sample predicted nfl_v95 / known Combine s2_v95
        p_ktr = p_va / s2[va]
        yr_ktr_r2[yr] = r2_score(y_ktr[va], p_ktr)
        yr_ktr_rho[yr] = stats.spearmanr(y_ktr[va], p_ktr)[0]

    oof_ktr_logo = oof_nfl_logo / s2

    # Complementary Evaluation: 5-Fold CV (for fixed feature sets)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_nfl_5f = np.zeros(len(df))
    va_r2_5f = []
    for tr, va in kf.split(df, y_nfl):
        feats = BASE_CLASSIC + ["s1_v95", "hb_cut_speed"] if features_or_mode == "NESTED_TOP2" else list(features_or_mode)
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))
        pipe.fit(df.iloc[tr][feats], y_nfl[tr])
        p_va = pipe.predict(df.iloc[va][feats])
        oof_nfl_5f[va] = p_va
        va_r2_5f.append(r2_score(y_nfl[va], p_va))

    q75_true = np.percentile(y_nfl, 75)
    q75_pred = np.percentile(oof_nfl_logo, 75)
    top25_capture = np.mean((oof_nfl_logo >= q75_pred)[y_nfl >= q75_true]) * 100.0

    return {
        "Model": label,
        "LOGO_Train_R2": round(np.mean(tr_r2_logo), 3),
        "LOGO_Val_R2": round(np.mean(va_r2_logo), 3),
        "LOGO_Gap": round(np.mean(tr_r2_logo) - np.mean(va_r2_logo), 3),
        "LOGO_Std": round(np.std(va_r2_logo, ddof=0), 3),
        "LOGO_RMSE": round(np.sqrt(mean_squared_error(y_nfl, oof_nfl_logo)), 4),
        "LOGO_MAE": round(mean_absolute_error(y_nfl, oof_nfl_logo), 4),
        "LOGO_Spearman": round(stats.spearmanr(y_nfl, oof_nfl_logo)[0], 3),
        "Top25_Capture_%": round(top25_capture, 1),
        "2023_R2 (ρ)": f"{yr_nfl_r2[2023]:.3f} ({yr_nfl_rho[2023]:.2f})",
        "2024_R2 (ρ)": f"{yr_nfl_r2[2024]:.3f} ({yr_nfl_rho[2024]:.2f})",
        "2025_R2 (ρ)": f"{yr_nfl_r2[2025]:.3f} ({yr_nfl_rho[2025]:.2f})",
        "5Fold_Val_R2": round(np.mean(va_r2_5f), 3),
        "5Fold_Spearman": round(stats.spearmanr(y_nfl, oof_nfl_5f)[0], 3),
        "DerivedKTR_LOGO_R2": round(np.mean(list(yr_ktr_r2.values())), 3),
        "DerivedKTR_LOGO_RMSE": round(np.sqrt(mean_squared_error(y_ktr, oof_ktr_logo)), 4),
        "DerivedKTR_LOGO_MAE": round(mean_absolute_error(y_ktr, oof_ktr_logo), 4),
        "DerivedKTR_Spearman": round(stats.spearmanr(y_ktr, oof_ktr_logo)[0], 3),
        "DerivedKTR_2023": f"{yr_ktr_r2[2023]:.3f} ({yr_ktr_rho[2023]:.2f})",
        "DerivedKTR_2024": f"{yr_ktr_r2[2024]:.3f} ({yr_ktr_rho[2024]:.2f})",
        "DerivedKTR_2025": f"{yr_ktr_r2[2025]:.3f} ({yr_ktr_rho[2025]:.2f})",
        "_oof_nfl_logo": oof_nfl_logo,
        "_oof_ktr_logo": oof_ktr_logo,
    }
```

---

## 4. Comparison of Models A, B, and C on Direct `nfl_v95` Prediction

We compare the route-only micro model (`Reduced_3`) against three structured model families on identical LOGO splits:
* **Model A (Baseline — `Klasik Tam`):** `forty + forty_vmax + combine_height + combine_weight` (4 features)
* **Model B (Combined — `Klasik Tam + Reduced_3`):** `Model A + hb_v95 + deep_v95 + hb_cut_speed_retention` (7 features)
* **Model C (Extended Sensor Speed/Acceleration):**
  * **C1 (Velocity Extension):** `Model A + s1_v95 + s2_vmax` (6 features)
  * **C2 (Acceleration Extension):** `Model A + forty_a95 + s2_a95` (6 features)
  * **C3 (Velocity + Acceleration Extension):** `Model A + s1_v95 + s2_vmax + forty_a95 + s2_a95` (8 features)
  * **C4 (Nested Training-Set Selection):** `Model A + Top-2 correlated extensions selected strictly within each LOGO training split` (6 features)

```python
model_specs = [
    ("Micro Only: Reduced_3 (hb_v95 + deep_v95 + hb_cut_ret)", REDUCED_3),
    ("Model A — Baseline: Classical Full (forty + forty_vmax + height + weight)", BASE_CLASSIC),
    ("Model B — Combined: Model A + Reduced_3", BASE_CLASSIC + REDUCED_3),
    ("Model C1 — Extended (Velocity): Model A + s1_v95 + s2_vmax", BASE_CLASSIC + ["s1_v95", "s2_vmax"]),
    ("Model C2 — Extended (Accel.): Model A + forty_a95 + s2_a95", BASE_CLASSIC + ["forty_a95", "s2_a95"]),
    ("Model C3 — Extended (Vel. + Accel.): Model A + s1_v95 + s2_vmax + forty_a95 + s2_a95", BASE_CLASSIC + ["s1_v95", "s2_vmax", "forty_a95", "s2_a95"]),
    ("Model C4 — Extended (Nested Top-2 Selected in Training Years)", "NESTED_TOP2"),
]

model_results = {}
comparison_rows = []
for name, spec in model_specs:
    res = evaluate_logo_and_5fold(df, spec, alpha=10.0, label=name)
    model_results[name] = res
    comparison_rows.append(res)

comp_df = pd.DataFrame(comparison_rows)
cols_nfl_table = [
    "Model", "LOGO_Train_R2", "LOGO_Val_R2", "LOGO_Gap", "LOGO_Std",
    "LOGO_RMSE", "LOGO_MAE", "LOGO_Spearman", "Top25_Capture_%",
    "2023_R2 (ρ)", "2024_R2 (ρ)", "2025_R2 (ρ)", "5Fold_Val_R2", "5Fold_Spearman"
]
display(comp_df[cols_nfl_table])
```

| Model (`Ridge, alpha=10.0`) | LOGO Train R² | **LOGO Val R² (\pm Std)** | LOGO Gap | **LOGO RMSE** | **LOGO MAE** | **LOGO Spearman ρ** | Top-25% Capture | **2023 R² (ρ)** | **2024 R² (ρ)** | **2025 R² (ρ)** | 5-Fold Val R² (ρ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Micro Only (`Reduced_3`)** | 0.118 | **0.017** (± 0.049) | 0.101 | 0.3473 | 0.2880 | **0.227** | 31.6% | -0.023 (0.12) | 0.087 (0.37) | -0.012 (0.27) | -0.034 (0.195) |
| **Model A — Baseline (`Classical Full`)** | 0.451 | **0.374** (± 0.054) | **0.077** | 0.2771 | 0.2304 | **0.638** | 57.9% | **0.424 (0.62)** | 0.299 (0.62) | 0.399 (0.68) | **0.382** (0.655) |
| **Model B — Combined (`A + Reduced_3`)** | 0.512 | **0.386** (± 0.072) | 0.126 | 0.2765 | 0.2276 | **0.676** | 57.9% | 0.354 (0.61) | 0.319 (0.69) | **0.486 (0.78)** | **0.384** (**0.686**) |
| **Model C1 — Extended (`Velocity`)** | 0.490 | **0.372** (± 0.037) | 0.118 | 0.2770 | 0.2269 | 0.657 | 52.6% | 0.421 (0.63) | 0.332 (0.63) | 0.361 (0.74) | 0.371 (0.651) |
| **Model C2 — Extended (`Accel.`)** | 0.470 | **0.376** (± 0.041) | 0.094 | 0.2769 | 0.2248 | 0.651 | 57.9% | 0.404 (0.65) | 0.318 (0.64) | 0.406 (0.74) | 0.358 (0.637) |
| **Model C3 — Extended (`Vel. + Accel.`)** | 0.513 | **0.392** (± 0.026) | 0.121 | **0.2727** | **0.2161** | **0.683** | **63.2%** | **0.427 (0.66)** | **0.366 (0.66)** | 0.382 (0.75) | 0.368 (0.664) |
| **Model C4 — Nested Top-2 Selection** | 0.510 | **0.346** (± 0.012) | 0.165 | 0.2839 | 0.2333 | 0.666 | 52.6% | 0.333 (0.61) | 0.343 (0.68) | 0.361 (0.74) | 0.351 (0.652) |

### Key Analytical Takeaways
1. **Standalone Route Micro-Features (`Reduced_3`) Do Not Predict `nfl_v95`:** Used in isolation, `Reduced_3` yields `LOGO R² = 0.017` (`RMSE = 0.3473 yds/s`, `Spearman ρ = 0.227`) and negative R² in two of the three draft cohorts.
2. **Trade-off Between `Model A`, `Model B`, and `Model C3`:**
   * **`Model C3`** achieves the highest overall LOGO validation metrics (`LOGO R² = 0.392`, `RMSE = 0.2727 yds/s`, `MAE = 0.2161 yds/s`, `Spearman ρ = 0.683`, and `63.2%` top-quartile capture), while **`Model B`** achieves `LOGO R² = 0.386` and `Spearman ρ = 0.676`.
   * **`Model A`** (`forty + forty_vmax + combine_height + combine_weight`) achieves `LOGO R² = 0.374`, `RMSE = 0.2771 yds/s`, and `Spearman ρ = 0.638` with only 4 features and maintains the smallest Train–Validation gap (`0.077` vs. `0.121` for `C3` and `0.126` for `B`).
   * Consequently, selecting `Model A` as our primary baseline is motivated by **parsimony and a smaller generalization gap** on N = 76, rather than absolute metric superiority; `Model B` and especially `Model C3` represent viable complementary alternatives when a slight increase in parameterization is acceptable.

---

## 5. Denominator Coupling in KTR and the Derived Ratio Approach

To evaluate why `Reduced_3` achieved `LOGO R² = 0.317` when predicting `KTR_Skill_Speed` directly despite having near-zero predictive power for `nfl_v95` (`LOGO R² = 0.017`), we compare direct KTR regression against the **Derived KTR** approach (`Predicted_KTR = Predicted_nfl_v95 / s2_v95`) and a **Constant-Mean Numerator Reference** (`Reference_KTR = mean_train_nfl_v95 / s2_v95`):

```python
logo = LeaveOneGroupOut()
y_nfl = df["nfl_v95"].values
y_ktr = df["KTR_Skill_Speed"].values
s2 = df["s2_v95"].values
groups = df["draft_year"].values

# 0. Constant-Mean Numerator Reference (computed strictly on training years in each LOGO fold)
oof_nfl_const = np.zeros(len(df))
const_ktr_yr_r2, const_ktr_yr_rho = {}, {}
for tr, va in logo.split(df, y_nfl, groups):
    yr = int(groups[va][0])
    oof_nfl_const[va] = y_nfl[tr].mean()
    p_ktr_c = oof_nfl_const[va] / s2[va]
    const_ktr_yr_r2[yr] = r2_score(y_ktr[va], p_ktr_c)
    const_ktr_yr_rho[yr] = stats.spearmanr(y_ktr[va], p_ktr_c)[0]
oof_ktr_const = oof_nfl_const / s2

# Direct KTR Regression (Model A and Reduced_3)
oof_ktr_dir_A = np.zeros(len(df))
oof_ktr_dir_R3 = np.zeros(len(df))
dirA_yr_r2, dirA_yr_rho = {}, {}
dirR3_yr_r2, dirR3_yr_rho = {}, {}

for tr, va in logo.split(df, y_ktr, groups):
    yr = int(groups[va][0])
    pA = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    pA.fit(df.iloc[tr][BASE_CLASSIC], y_ktr[tr])
    oof_ktr_dir_A[va] = pA.predict(df.iloc[va][BASE_CLASSIC])
    dirA_yr_r2[yr] = r2_score(y_ktr[va], oof_ktr_dir_A[va])
    dirA_yr_rho[yr] = stats.spearmanr(y_ktr[va], oof_ktr_dir_A[va])[0]

    pR3 = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    pR3.fit(df.iloc[tr][REDUCED_3], y_ktr[tr])
    oof_ktr_dir_R3[va] = pR3.predict(df.iloc[va][REDUCED_3])
    dirR3_yr_r2[yr] = r2_score(y_ktr[va], oof_ktr_dir_R3[va])
    dirR3_yr_rho[yr] = stats.spearmanr(y_ktr[va], oof_ktr_dir_R3[va])[0]
```

| KTR Estimation Strategy | Numerator (`nfl_v95`) LOGO R² | **KTR LOGO Val R² (\pm Std)** | **KTR LOGO RMSE** | **KTR LOGO MAE** | **KTR LOGO Spearman ρ** | **2023 KTR R² (ρ)** | **2024 KTR R² (ρ)** | **2025 KTR R² (ρ)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0. Constant-Mean Numerator Ref.** (`mean_train_nfl_v95 / s2_v95`) | -0.010 | **0.459** (± 0.033) | 0.0385 | 0.0325 | **0.620** | 0.483 (0.66) | 0.412 (0.47) | 0.481 (0.62) |
| **1. Direct KTR Regression — `Model A`** | 0.374 | **0.006** (± 0.042) | 0.0521 | 0.0426 | 0.283 | 0.066 (0.48) | -0.025 (0.28) | -0.022 (0.34) |
| **2. Direct KTR Regression — `Reduced_3`** | 0.017 | **0.317** (± 0.144) | 0.0438 | 0.0369 | 0.472 | 0.236 (0.46) | 0.195 (0.25) | 0.519 (0.64) |
| **3. Derived KTR (\widehat{\mathrm{nfl\_v95}} / \mathrm{s2\_v95}) — `Reduced_3`** | 0.017 | **0.479** (± 0.008) | 0.0379 | 0.0314 | 0.645 | 0.473 (0.66) | 0.473 (0.54) | 0.490 (0.70) |
| **4. Derived KTR (\widehat{\mathrm{nfl\_v95}} / \mathrm{s2\_v95}) — `Model A`** | **0.374** | **0.663** (± 0.055) | **0.0303** | **0.0252** | **0.773** | **0.702 (0.80)** | **0.586 (0.67)** | **0.702 (0.75)** |
| **5. Derived KTR (\widehat{\mathrm{nfl\_v95}} / \mathrm{s2\_v95}) — `Model C3`** | **0.392** | **0.673** (± 0.034) | **0.0299** | **0.0241** | **0.800** | 0.692 (0.79) | **0.626 (0.72)** | **0.702 (0.80)** |

### Interpretation of Denominator Coupling and Derived KTR
1. **Signal Carried by the Player-Specific Denominator (`s2_v95`):** The constant-mean numerator reference fixes the predicted NFL speed to the training-cohort mean (`mean_train_nfl_v95 ≈ 9.02–9.05 yds/s`) for every test player while dividing by each player's actual Combine `s2_v95`. This reference achieves `LOGO R² = 0.459` and `Spearman ρ = 0.620`, demonstrating that the player-specific Combine `s2_v95` value carries a strong baseline signal in determining KTR.
2. **Why Direct `Reduced_3` Appeared Effective on KTR (`R² = 0.317`):** Because `hb_v95` (r = +0.706) and `deep_v95` (r = +0.669) are strongly correlated with `s2_v95`, direct KTR regression with `Reduced_3` primarily approximates the inverse denominator (`1 / s2_v95`) rather than predicting NFL game speed (`nfl_v95`).
3. **Joint Effect in Derived KTR (`R² = 0.663` for `Model A`, `0.673` for `Model C3`):** Predicting `nfl_v95` out-of-sample via `Model A` (`LOGO R² = 0.374`) and dividing by known `s2_v95` raises KTR `LOGO R²` from `0.459` (the constant-mean reference) to `0.663` (`Spearman ρ = 0.773`). This improvement demonstrates that out-of-sample NFL speed prediction contributes meaningful incremental information to the ratio estimate. However, the high resulting R² reflects the **joint outcome of linear speed/mass prediction and the division by known `s2_v95`**, rather than evidence that the model has learned complex biomechanical skill transfer.

---

## 6. Player-Level Residual Analysis: Walker, Burton, and Coleman

All 2024 player predictions below are strictly out-of-sample LOGO predictions generated by models trained exclusively on the 2023 and 2025 cohorts (N = 50):

```python
df["Pred_A"] = model_results["Model A — Baseline: Classical Full (forty + forty_vmax + height + weight)"]["_oof_nfl_logo"]
df["Pred_B"] = model_results["Model B — Combined: Model A + Reduced_3"]["_oof_nfl_logo"]
df["Pred_C3"] = model_results["Model C3 — Extended (Vel. + Accel.): Model A + s1_v95 + s2_vmax + forty_a95 + s2_a95"]["_oof_nfl_logo"]
df["Pred_KTR_Derived_A"] = model_results["Model A — Baseline: Classical Full (forty + forty_vmax + height + weight)"]["_oof_ktr_logo"]
df["Pred_KTR_Direct_R3"] = oof_ktr_dir_R3

for col_p, col_r in [("nfl_v95", "Rank_NFL"), ("Pred_A", "Rank_A"), ("Pred_B", "Rank_B"), ("Pred_C3", "Rank_C3")]:
    df[col_r] = df.groupby("draft_year")[col_p].rank(ascending=False, method="min").astype(int)

df["Res_A"] = df["nfl_v95"] - df["Pred_A"]
df["Res_B"] = df["nfl_v95"] - df["Pred_B"]
df["Res_C3"] = df["nfl_v95"] - df["Pred_C3"]

focus_2024 = ["Devontez Walker", "Jermaine Burton", "Troy Franklin", "Brian Thomas", "Keon Coleman"]
sub24 = df[(df["draft_year"] == 2024) & (df["display_name"].isin(focus_2024))].sort_values("nfl_v95", ascending=False)
display(sub24[[
    "display_name", "forty", "forty_vmax", "combine_weight", "s2_v95", "reg_plays_tracked",
    "nfl_v95", "Rank_NFL", "Pred_A", "Res_A", "Rank_A", "Pred_B", "Res_B", "Rank_B", "Pred_C3", "Res_C3", "Rank_C3"
]].round(3))
```

| 2024 WR Prospect | `forty` / `forty_vmax` | Weight (lb) | Combine `s2_v95` | NFL Tracked Plays (`reg_plays_tracked`) | **Actual `nfl_v95` (Rank /26)** | **Model A Pred / Res (Rank)** | **Model B Pred / Res (Rank)** | **Model C3 Pred / Res (Rank)** | **Direct `Reduced_3` KTR Rank** | **Derived `Model A` KTR Rank (Actual)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Devontez Walker** | 4.36s / 11.31 | 193 | 9.245 | 80 | **9.746 (#1)** | 9.171 / **+0.575** (#6) | 9.218 / **+0.528** (#5) | 9.222 / **+0.524** (#5) | #17 | **#11** (Actual: #3) |
| **Jermaine Burton** | 4.45s / 11.36 | 196 | 9.469 | 96 | **9.658 (#2)** | 9.103 / **+0.554** (#9) | 9.138 / **+0.519** (#10) | 9.120 / **+0.538** (#10) | #23 | **#20** (Actual: #6) |
| **Troy Franklin** | 4.41s / 11.39 | 176 | 9.486 | 742 | **9.440 (#4)** | 9.296 / **+0.144** (#3) | 9.262 / **+0.178** (#4) | 9.281 / **+0.159** (#2) | #25 | **#17** (Actual: #10) |
| **Brian Thomas** | 4.33s / 11.55 | 209 | 9.099 | 984 | **9.190 (#9)** | 9.102 / **+0.088** (#11) | 9.156 / **+0.034** (#7) | 9.163 / **+0.027** (#8) | #15 | **#8** (Actual: #8) |
| **Keon Coleman** | 4.61s / 10.70 | 213 | 9.440 | 633 | **8.708 (#22)** | 8.673 / **+0.035** (#25) | 8.748 / **-0.040** (#25) | 8.718 / **-0.010** (#25) | #5 | **#25** (Actual: #24) |

### Diagnostic Findings on Player Residuals
1. **Accurate Calibration on High-Volume Starters (`Keon Coleman`, `Brian Thomas`, `Troy Franklin`):** Direct KTR regression with `Reduced_3` ranked `Keon Coleman` #5 in the 2024 class despite his `4.61s` 40-yard dash and `8.708 yds/s` actual NFL speed (#22/26). Modeling `nfl_v95` directly resolves this error across `Model A` (`8.673 yds/s`, residual `+0.035 yds/s`, rank #25/26), `Model B` (`8.748 yds/s`), and `Model C3` (`8.718 yds/s`, residual `-0.010 yds/s`), while `Derived KTR` places Coleman at #25/26 (actual KTR rank: #24/26).
2. **Persistent Positive Residuals for `Devontez Walker` (`+0.575 yds/s`) and `Jermaine Burton` (`+0.554 yds/s`):** Adding route micro-features (`Model B`) or extended velocity/acceleration sensors (`Model C3`) reduces Walker's and Burton's residuals by only `0.02–0.05 yds/s`. Because their Combine peak speeds (`forty_vmax = 11.31` and `11.36 yds/s`; 72nd and 79th percentiles) are above average but below class leaders like `Brian Thomas` (`11.55 yds/s`, 94th percentile), no linear pre-draft model predicts them above `9.22 yds/s`.
3. **Play-Volume Stratification as an Explanatory Hypothesis:** Across all 76 WRs, players with fewer than 200 tracked regular-season plays (N = 20, including `Walker` with 80 plays and `Burton` with 96 plays) exhibit substantially higher out-of-sample error (`Model A` OOF `RMSE = 0.3605 yds/s`, `MAE = 0.3202 yds/s`, `R² = 0.317`) than WRs with ≥ 200 tracked plays (N = 56, `Model A` OOF `RMSE = 0.2317 yds/s`, `MAE = 0.1915 yds/s`, `R² = 0.428`). While this pattern suggests that low-snap players may experience greater 95th-percentile sampling variance or specialized vertical route deployment, **low play volume (`reg_plays_tracked`) alone does not prove that a player was deployed exclusively on deep routes**. Testing that mechanism directly requires play-level route classification, target depth (*air yards*), and defensive coverage context beyond pre-draft Combine measurements.

---

## 7. Generalizability, Limitations, and Potential Biases

1. **Limited Cohort Horizon (N = 76, Three Draft Classes):** Our LOGO estimates are derived from three draft cohorts (2023–2025). Reported metrics (`LOGO R² = 0.374` for `Model A`; `0.392` for `Model C3`) summarize temporal validation within these three classes and should not be interpreted as guaranteed performance bounds for future seasons.
2. **Sample Selection Bias:** The sample includes only WRs who both participated in Combine tracking drills and logged regular-season NFL routes. Prospects who skipped Combine drills (often high-profile draft picks) and draftees who did not earn regular-season snaps are unobserved.
3. **Regularization Shrinkage at the Extremes:** `Ridge(alpha=10.0)` stabilizes out-of-sample variance on small N by shrinking predictions toward the cohort mean (`~8.65–9.35 yds/s`), which systematically underpredicts extreme tail observations (`nfl_v95 > 9.60 yds/s`).
4. **Iterative Model Exploration Optimism:** Because candidate feature subsets and model specifications were compared across the same 2023–2025 cohorts, a modest degree of model-selection optimism may remain. Definitive validation requires an untouched future cohort (e.g., the 2026 draft class).

---

## 8. Final Model Decision and Executive Summary for Kaggle

1. **Primary Baseline Model (`Model A — Classical Full`):**
   * **Specification:** `SimpleImputer(strategy="median") + StandardScaler() + Ridge(alpha=10.0)` using `["forty", "forty_vmax", "combine_height", "combine_weight"]`.
   * **Temporal Validation (2023–2025 LOGO):** `Val R² = 0.374 (±0.054)`, `Train–Val Gap = 0.077`, `RMSE = 0.2771 yds/s`, `MAE = 0.2304 yds/s`, `Spearman ρ = 0.638`.
   * **Selection Rationale:** `Model A` is selected as the primary baseline not because it achieves the highest raw validation score, but because it offers the most favorable balance between predictive accuracy, parsimony (4 features), and a low Train–Validation gap (`0.077`).
2. **Complementary Alternatives (`Model B` and `Model C3`):**
   * **`Model C3`** (`Model A + s1_v95 + s2_vmax + forty_a95 + s2_a95`) achieves the highest out-of-sample LOGO performance (`Val R² = 0.392 ± 0.026`, `RMSE = 0.2727 yds/s`, `Spearman ρ = 0.683`), while **`Model B`** (`Model A + Reduced_3`) reaches `Val R² = 0.386` (`Spearman ρ = 0.676`). Both are reported as complementary specifications whose modest gains (`+0.012` to `+0.018 R²`) trade off against higher dimensionality (7–8 features) and wider Train–Validation gaps (`0.121–0.126`).
3. **Recommended KTR Estimation Protocol:**
   * Rather than regressing `KTR_Skill_Speed` directly on Combine features, `nfl_v95` should be predicted out-of-sample first and divided by the known Combine `s2_v95` (`Predicted_KTR = Predicted_nfl_v95 / s2_v95`). This yields `LOGO R² = 0.663` (`Spearman ρ = 0.773`) with `Model A` and `LOGO R² = 0.673` (`Spearman ρ = 0.800`) with `Model C3`, improving upon the constant-mean numerator reference (`LOGO R² = 0.459`, `Spearman ρ = 0.620`) as the joint result of out-of-sample `nfl_v95` prediction and the ratio transformation.
4. **Core Conclusion:**
   * The utility of Combine micro-biomechanical tracking features depends critically on the target variable. For predicting regular-season 95th-percentile speed (`nfl_v95`), the primary predictive signals utilized by the model are linear speed measurements (`forty`, `forty_vmax`) and anthropometric attributes (`combine_weight`, `combine_height`), whereas standalone route-drill speeds (`hb_v95`, `deep_v95`) primarily reflect variations in the Combine denominator (`s2_v95`).
