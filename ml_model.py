# ============================================================
# OpenPowerlifting — Wilks Score Prediction Model
# ml_model.py
#
# Imports the cleaned dataset from analysis.py and builds:
#   Step 1   — Load data via analysis.load_data()
#   Step 1.5 — Athlete history features (career Wilks stats)
#   Step 2   — Feature preparation & train/test split
#   Stage 1  — Linear Regression baseline (sklearn Pipeline + OHE)
#   Stage 2  — Random Forest Regressor
#   Stage 2b — LightGBM Regressor (history features, best model)
#   Stage 3  — 5-Fold Cross-Validation (Random Forest)
#   Stage 4  — Evaluation charts
#   Stage 5  — Model persistence (5 joblib files)
#   Stage 6  — predict_wilks / predict_wilks_lr / predict_wilks_lgb demo
#
# HOW TO RUN:
#   Make sure analysis.py, meets.csv, and openpowerlifting.csv
#   are all in the same folder, then run:
#       python ml_model.py
#
#   After running, the trained models are saved as:
#       wilks_rf_model.joblib      wilks_encoders.joblib
#       wilks_lr_pipeline.joblib
#       wilks_lgb_model.joblib     wilks_lgb_encoders.joblib
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import mean_absolute_error, r2_score, mean_squared_error

from analysis import load_data


# ─────────────────────────────────────────────────────────────
# MODULE-LEVEL CONSTANTS
# ─────────────────────────────────────────────────────────────

FEATURES = ["Sex", "Equipment", "AgeGroup", "BodyweightKg", "WeightClassKg"]
TARGET   = "Wilks"

# AgeGroup is genuinely ordinal — explicit integer mapping preserves
# the biological/competitive order instead of letting LabelEncoder
# assign arbitrary numbers.
AGE_GROUP_ORDER = {
    "Sub-Junior (<18)": 0,
    "Junior (18-23)"  : 1,
    "Open (24-39)"    : 2,
    "Master 40-49"    : 3,
    "Master 50-59"    : 4,
    "Master 60+"      : 5,
}

MODEL_PATH       = "wilks_rf_model.joblib"
ENCODERS_PATH    = "wilks_encoders.joblib"
LR_PIPELINE_PATH = "wilks_lr_pipeline.joblib"
LGB_MODEL_PATH   = "wilks_lgb_model.joblib"
LGB_ENCODERS_PATH = "wilks_lgb_encoders.joblib"

FEATURES_HIST = FEATURES + [
    "Age",                        # continuous age — more granular than AgeGroup bins
    "Year",                       # competition year — performance trends over time
    "athlete_prev_count",
    "is_first_comp",
    "athlete_prev_wilks_mean",
    "athlete_prev_wilks_last",
    "athlete_prev_wilks_std",
    "athlete_prev_wilks_best",
    "athlete_prev_wilks_recent3", # mean of last 3 comps — recent form beats career avg
]


# ─────────────────────────────────────────────────────────────
# predict_wilks()
# Defined at module level so it can be imported by other scripts.
# Pass model/encoders directly (after training) or leave them as
# None to auto-load from the saved .joblib files.
# ─────────────────────────────────────────────────────────────

def predict_wilks(sex, equipment, age_group, bodyweight_kg, weight_class_kg,
                  model=None, encoders=None):
    """
    Predicts the Wilks score for a new athlete using the trained
    Random Forest model.

    Parameters:
        sex             : "M" or "F"
        equipment       : "Raw", "Wraps", "Single-ply", or "Multi-ply"
        age_group       : "Sub-Junior (<18)", "Junior (18-23)", "Open (24-39)",
                          "Master 40-49", "Master 50-59", or "Master 60+"
        bodyweight_kg   : athlete bodyweight in kilograms (float)
        weight_class_kg : weight class ceiling in kilograms (float)
        model           : trained RandomForestRegressor  (optional —
                          loads wilks_rf_model.joblib if not provided)
        encoders        : dict of fitted LabelEncoders   (optional —
                          loads wilks_encoders.joblib if not provided)

    Returns:
        Predicted Wilks score (float)
    """
    if model is None:
        model = joblib.load(MODEL_PATH)
    if encoders is None:
        encoders = joblib.load(ENCODERS_PATH)

    sex_enc   = encoders["Sex"].transform([sex])[0]
    equip_enc = encoders["Equipment"].transform([equipment])[0]
    age_enc   = AGE_GROUP_ORDER[age_group]

    features = np.array([[sex_enc, equip_enc, age_enc,
                          bodyweight_kg, weight_class_kg]])
    return round(model.predict(features)[0], 1)


# ─────────────────────────────────────────────────────────────
# predict_wilks_lr()
# Linear Regression variant — loads the full sklearn Pipeline
# (ColumnTransformer + LinearRegression) saved during training.
# ─────────────────────────────────────────────────────────────

def predict_wilks_lr(sex, equipment, age_group, bodyweight_kg, weight_class_kg,
                     lr_pipeline=None):
    """
    Predicts the Wilks score for a new athlete using the trained
    Linear Regression pipeline.

    Parameters: same as predict_wilks().
    Returns:    Predicted Wilks score (float)
    """
    if lr_pipeline is None:
        lr_pipeline = joblib.load(LR_PIPELINE_PATH)

    row = pd.DataFrame([{
        "Sex"          : sex,
        "Equipment"    : equipment,
        "AgeGroup"     : AGE_GROUP_ORDER[age_group],
        "BodyweightKg" : float(bodyweight_kg),
        "WeightClassKg": float(weight_class_kg),
    }])[FEATURES]
    return round(lr_pipeline.predict(row)[0], 1)


# ─────────────────────────────────────────────────────────────
# predict_wilks_lgb()
# LightGBM variant — includes athlete history features.
# All history args default to first-time-competitor values so
# the function works even without prior competition data.
# ─────────────────────────────────────────────────────────────

def predict_wilks_lgb(sex, equipment, age_group, bodyweight_kg, weight_class_kg,
                      age=None, year=2024,
                      athlete_prev_count=0, is_first_comp=1,
                      athlete_prev_wilks_mean=None, athlete_prev_wilks_last=None,
                      athlete_prev_wilks_std=None, athlete_prev_wilks_best=None,
                      athlete_prev_wilks_recent3=None,
                      model=None, encoders=None):
    """
    Predicts Wilks score using the LightGBM model with athlete history.

    Parameters: same as predict_wilks() for the base 5 features, plus:
        age                       : float — raw age in years (None = unknown)
        year                      : int   — competition year
        athlete_prev_count        : int   — prior competition count (0 = debut)
        is_first_comp             : int   — 1 if debut, else 0
        athlete_prev_wilks_mean   : float — career avg Wilks     (None = unknown)
        athlete_prev_wilks_last   : float — most recent Wilks    (None = unknown)
        athlete_prev_wilks_std    : float — Wilks std dev        (None = unknown)
        athlete_prev_wilks_best   : float — career peak Wilks    (None = unknown)
        athlete_prev_wilks_recent3: float — mean of last 3 comps (None = unknown)

    Returns: Predicted Wilks score (float)
    """
    if model is None:
        model = joblib.load(LGB_MODEL_PATH)
    if encoders is None:
        encoders = joblib.load(LGB_ENCODERS_PATH)

    sex_enc   = encoders["Sex"].transform([sex])[0]
    equip_enc = encoders["Equipment"].transform([equipment])[0]
    age_enc   = AGE_GROUP_ORDER[age_group]

    row = np.array([[sex_enc, equip_enc, age_enc, bodyweight_kg, weight_class_kg,
                     age, year,
                     athlete_prev_count, is_first_comp,
                     athlete_prev_wilks_mean, athlete_prev_wilks_last,
                     athlete_prev_wilks_std, athlete_prev_wilks_best,
                     athlete_prev_wilks_recent3]],
                   dtype=float)
    return round(model.predict(row)[0], 1)


# ─────────────────────────────────────────────────────────────
# MAIN — only runs when executed directly, NOT when imported
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── STEP 1 — Load Data ───────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Loading cleaned data from analysis.py")
    print("=" * 60)

    df, valid = load_data(verbose=False)
    print(f"Full dataset  : {len(df):,} rows")
    print(f"Valid entries : {len(valid):,} rows")


    # ── STEP 1.5 — Athlete History Features ─────────────────
    print("\n" + "=" * 60)
    print("STEP 1.5 — Computing athlete history features")
    print("=" * 60)

    # Sort by Name then Date so .shift(1) gives the immediately prior
    # competition for each athlete. mergesort preserves original order
    # on same-date ties (rare but possible in multi-event meets).
    valid = valid.sort_values(['Name', 'Date'], kind='mergesort').copy()

    valid['athlete_prev_count'] = valid.groupby('Name').cumcount()
    valid['is_first_comp']      = (valid['athlete_prev_count'] == 0).astype(int)

    # Helper functions used with .transform() — each returns a Series
    # built from only the rows that came BEFORE the current row.
    def _lag_mean(s): return s.shift(1).expanding().mean()
    def _lag_std(s):  return s.shift(1).expanding().std()
    def _lag_max(s):  return s.shift(1).expanding().max()

    valid['athlete_prev_wilks_mean'] = (
        valid.groupby('Name')['Wilks'].transform(_lag_mean)
    )
    valid['athlete_prev_wilks_last'] = (
        valid.groupby('Name')['Wilks'].transform(lambda s: s.shift(1))
    )
    valid['athlete_prev_wilks_std']  = (
        valid.groupby('Name')['Wilks'].transform(_lag_std)
    )
    valid['athlete_prev_wilks_best'] = (
        valid.groupby('Name')['Wilks'].transform(_lag_max)
    )
    valid['athlete_prev_wilks_recent3'] = (
        valid.groupby('Name')['Wilks']
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )

    first_timers = int(valid['is_first_comp'].sum())
    print(f"First-time competitors : {first_timers:,}  ({first_timers/len(valid)*100:.1f}%)")
    print(f"Experienced athletes   : {len(valid) - first_timers:,}")

    # ── STEP 2 — Feature Preparation ────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Feature preparation")
    print("=" * 60)

    ml_df = valid[FEATURES + [TARGET]].copy()

    # WeightClassKg contains strings like "90+" — strip the "+" and convert
    ml_df["WeightClassKg"] = (
        ml_df["WeightClassKg"]
        .astype(str)
        .str.replace("+", "", regex=False)
        .replace("nan", float("nan"))
    )
    ml_df["WeightClassKg"] = pd.to_numeric(ml_df["WeightClassKg"], errors="coerce")
    ml_df["AgeGroup"]      = ml_df["AgeGroup"].replace("Unknown", float("nan"))
    ml_df = ml_df.dropna()

    print(f"Rows after dropping nulls : {len(ml_df):,}")
    print(f"Wilks range               : {ml_df[TARGET].min():.1f} → {ml_df[TARGET].max():.1f}")
    print(f"Wilks mean ± std          : {ml_df[TARGET].mean():.1f} ± {ml_df[TARGET].std():.1f}")

    # ── Target distribution plot ─────────────────────────────
    # Always inspect the target variable before modelling.
    # A skewed distribution may require log-transform.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ml_df[TARGET], bins=100, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(ml_df[TARGET].mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"Mean ({ml_df[TARGET].mean():.1f})")
    ax.set_xlabel("Wilks Score")
    ax.set_ylabel("Number of Athletes")
    ax.set_title("Distribution of Wilks Scores (Target Variable)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("ml_wilks_distribution.png", dpi=120)
    plt.close()
    print("Saved → ml_wilks_distribution.png")

    # ── Preserve raw strings for the LR pipeline ────────────────
    # LR needs OHE on Equipment (nominal) — LabelEncoder would impose
    # a false ordinal relationship that Linear Regression would exploit
    # as a continuous signal. AgeGroup is already ordinal-mapped here
    # so the ColumnTransformer can scale it alongside the numerics.
    ml_df_raw = ml_df.copy()
    ml_df_raw["AgeGroup"] = ml_df_raw["AgeGroup"].map(AGE_GROUP_ORDER)

    # ── Encoding for Random Forest ───────────────────────────────
    # LabelEncoder is fine for tree-based models — splits don't imply
    # any ordering between the integer codes.
    encoders = {}
    for col in ["Sex", "Equipment"]:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col])
        encoders[col] = le
        print(f"  {col} encoded : {dict(zip(le.classes_, le.transform(le.classes_)))}")

    ml_df["AgeGroup"] = ml_df["AgeGroup"].map(AGE_GROUP_ORDER)
    print(f"  AgeGroup ordinal map : {AGE_GROUP_ORDER}")

    X = ml_df[FEATURES].values
    y = ml_df[TARGET].values

    # Index-based split so LR and RF operate on identical folds
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=42
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Raw DataFrames for LR (Sex/Equipment still as strings)
    X_raw_train = ml_df_raw[FEATURES].iloc[train_idx].reset_index(drop=True)
    X_raw_test  = ml_df_raw[FEATURES].iloc[test_idx].reset_index(drop=True)

    print(f"\nTrain set : {len(X_train):,} rows")
    print(f"Test set  : {len(X_test):,} rows")

    # ── LGB feature matrix (FEATURES_HIST — 15 columns) ─────────
    # dropna only on the base 5 features + TARGET; Age/history NaNs
    # are kept — LGB handles them natively via its NaN split logic.
    ml_df_lgb = valid[FEATURES_HIST + [TARGET]].copy()
    ml_df_lgb["WeightClassKg"] = (
        ml_df_lgb["WeightClassKg"]
        .astype(str).str.replace("+", "", regex=False).replace("nan", float("nan"))
    )
    ml_df_lgb["WeightClassKg"] = pd.to_numeric(ml_df_lgb["WeightClassKg"], errors="coerce")
    ml_df_lgb["AgeGroup"]      = ml_df_lgb["AgeGroup"].replace("Unknown", float("nan"))
    ml_df_lgb = ml_df_lgb.dropna(subset=FEATURES + [TARGET])

    ml_df_lgb_enc = ml_df_lgb.copy()
    for col in ["Sex", "Equipment"]:
        ml_df_lgb_enc[col] = encoders[col].transform(ml_df_lgb_enc[col])
    ml_df_lgb_enc["AgeGroup"] = ml_df_lgb_enc["AgeGroup"].map(AGE_GROUP_ORDER)

    X_lgb = ml_df_lgb_enc[FEATURES_HIST].values
    y_lgb = ml_df_lgb_enc[TARGET].values

    lgb_train_idx, lgb_test_idx = train_test_split(
        np.arange(len(X_lgb)), test_size=0.2, random_state=42
    )
    X_lgb_train, X_lgb_test = X_lgb[lgb_train_idx], X_lgb[lgb_test_idx]
    y_lgb_train, y_lgb_test = y_lgb[lgb_train_idx], y_lgb[lgb_test_idx]
    print(f"LGB train : {len(X_lgb_train):,} rows")
    print(f"LGB test  : {len(X_lgb_test):,} rows")


    # ── STAGE 1 — Linear Regression (baseline, proper preprocessing) ──
    print("\n" + "=" * 60)
    print("STAGE 1 — Linear Regression (baseline)")
    print("=" * 60)

    # Equipment → OneHotEncoder(drop='first'): avoids the fictitious
    # ordinal relationship that a single integer column would force LR
    # to model with a single coefficient across all equipment types.
    # Sex → OneHotEncoder(drop='if_binary'): binary flag, same result
    # as LabelEncoder but keeps the pipeline self-contained.
    # Numerics + AgeGroup ordinal → StandardScaler: puts all continuous
    # inputs on the same scale so coefficients are directly comparable.
    _lr_pre = ColumnTransformer([
        ("num",   StandardScaler(),
                  ["BodyweightKg", "WeightClassKg", "AgeGroup"]),
        ("sex",   OneHotEncoder(drop="if_binary",  sparse_output=False), ["Sex"]),
        ("equip", OneHotEncoder(drop="first",       sparse_output=False), ["Equipment"]),
    ], remainder="drop")

    lr_pipeline = Pipeline([
        ("prep", _lr_pre),
        ("lr",   LinearRegression()),
    ])

    lr_pipeline.fit(X_raw_train, y_train)
    lr_preds = lr_pipeline.predict(X_raw_test)

    lr_mae  = mean_absolute_error(y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2   = r2_score(y_test, lr_preds)

    print(f"MAE  (Mean Absolute Error)       : {lr_mae:.2f} Wilks pts")
    print(f"RMSE (Root Mean Squared Error)   : {lr_rmse:.2f} Wilks pts")
    print(f"R²   (Explained Variance)        : {lr_r2:.4f}  ({lr_r2*100:.1f}%)")

    # Extract named coefficients (numerics are in standardised units)
    prep_step = lr_pipeline.named_steps["prep"]
    lr_step   = lr_pipeline.named_steps["lr"]
    feat_names = [n.split("__", 1)[-1]
                  for n in prep_step.get_feature_names_out()]
    coef_df = pd.DataFrame({
        "Feature"    : feat_names,
        "Coefficient": lr_step.coef_.round(3),
    }).sort_values("Coefficient", key=abs, ascending=False)
    print("\nLinear Regression Coefficients (numeric features standardised):")
    print(coef_df.to_string(index=False))


    # ── STAGE 2 — Random Forest Regressor ───────────────────
    print("\n" + "=" * 60)
    print("STAGE 2 — Random Forest Regressor (main model)")
    print("=" * 60)
    print("Training... (this may take 30–60 seconds on large data)")

    rf = RandomForestRegressor(
        n_estimators=100,     # number of decision trees
        max_depth=12,         # max tree depth — limits overfitting
        min_samples_leaf=10,  # minimum samples per leaf node
        random_state=42,      # fixed seed for reproducibility
        n_jobs=-1             # use all available CPU cores
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    rf_mae  = mean_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2   = r2_score(y_test, rf_preds)

    print(f"MAE  (Mean Absolute Error)       : {rf_mae:.2f} Wilks pts")
    print(f"RMSE (Root Mean Squared Error)   : {rf_rmse:.2f} Wilks pts")
    print(f"R²   (Explained Variance)        : {rf_r2:.4f}  ({rf_r2*100:.1f}%)")
    print(f"\nImprovement over Linear Regression:")
    print(f"  MAE  reduced by : {lr_mae  - rf_mae:.2f}  Wilks pts")
    print(f"  RMSE reduced by : {lr_rmse - rf_rmse:.2f}  Wilks pts")
    print(f"  R²   gained     : {rf_r2   - lr_r2:.4f}")

    importance_df = pd.DataFrame({
        "Feature"   : FEATURES,
        "Importance": rf.feature_importances_.round(4)
    }).sort_values("Importance", ascending=False)
    print("\nRandom Forest Feature Importance:")
    print(importance_df.to_string(index=False))


    # ── STAGE 2b — LightGBM (history features) ──────────────
    print("\n" + "=" * 60)
    print("STAGE 2b — LightGBM Regressor (11 features incl. history)")
    print("=" * 60)
    print("Training...")

    # categorical_feature=[0, 1] marks Sex and Equipment as unordered
    # nominal. AgeGroup stays numeric (ordinal 0–5) so threshold splits
    # respect the biological order rather than grouping categories.
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,       # more trees to complement the lower learning rate
        learning_rate=0.03,      # slower shrinkage → less variance, better generalisation
        max_depth=6,
        num_leaves=127,          # richer splits — captures deeper history interactions
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(X_lgb_train, y_lgb_train, categorical_feature=[0, 1])
    lgb_preds = lgb_model.predict(X_lgb_test)

    lgb_mae  = mean_absolute_error(y_lgb_test, lgb_preds)
    lgb_rmse = np.sqrt(mean_squared_error(y_lgb_test, lgb_preds))
    lgb_r2   = r2_score(y_lgb_test, lgb_preds)

    print(f"MAE  (Mean Absolute Error)       : {lgb_mae:.2f} Wilks pts")
    print(f"RMSE (Root Mean Squared Error)   : {lgb_rmse:.2f} Wilks pts")
    print(f"R²   (Explained Variance)        : {lgb_r2:.4f}  ({lgb_r2*100:.1f}%)")
    print(f"\nImprovement over Random Forest:")
    print(f"  MAE  reduced by : {rf_mae  - lgb_mae:.2f}  Wilks pts")
    print(f"  RMSE reduced by : {rf_rmse - lgb_rmse:.2f}  Wilks pts")
    print(f"  R²   gained     : {lgb_r2  - rf_r2:.4f}")

    lgb_importance_df = pd.DataFrame({
        "Feature"   : FEATURES_HIST,
        "Importance": lgb_model.feature_importances_,
    }).sort_values("Importance", ascending=False)
    print("\nLightGBM Feature Importance:")
    print(lgb_importance_df.to_string(index=False))

    # ── STAGE 3 — 5-Fold Cross-Validation ───────────────────
    print("\n" + "=" * 60)
    print("STAGE 3 — 5-Fold Cross-Validation (Random Forest)")
    print("=" * 60)
    # CV on a 30k sample keeps runtime reasonable while still being
    # statistically meaningful. Stable scores across folds confirm
    # the model generalises and isn't just memorising the train split.
    print("Running on a 30,000-row sample to keep it fast...")

    sample_size = min(30_000, len(X))
    rng = np.random.default_rng(42)
    cv_idx = rng.choice(len(X), size=sample_size, replace=False)

    cv_scores = cross_val_score(
        RandomForestRegressor(
            n_estimators=50, max_depth=12,
            min_samples_leaf=10, random_state=42, n_jobs=-1
        ),
        X[cv_idx], y[cv_idx],
        cv=5, scoring="r2"
    )
    print(f"CV R² per fold : {cv_scores.round(4)}")
    print(f"Mean R²        : {cv_scores.mean():.4f}  ±  {cv_scores.std():.4f}")
    print("Stable scores across folds = model generalises well.")


    # ── STAGE 4 — Evaluation Charts ─────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 4 — Generating evaluation charts")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Wilks Score Prediction — Model Evaluation", fontsize=13, fontweight="bold")

    # Chart A — MAE & RMSE comparison (LR / RF / LGB)
    ax = axes[0]
    x_pos = [0, 1, 2, 4, 5, 6]
    vals   = [lr_mae, rf_mae, lgb_mae, lr_rmse, rf_rmse, lgb_rmse]
    colors = ["#AAAAAA", "#4C72B0", "#1A3A6B",
              "#AAAAAA", "#4C72B0", "#1A3A6B"]
    bars = ax.bar(x_pos, vals, color=colors, width=0.7, edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["LR\nMAE", "RF\nMAE", "LGB\nMAE",
                         "LR\nRMSE", "RF\nRMSE", "LGB\nRMSE"], fontsize=8)
    ax.set_title("Error Metrics Comparison\n(lower is better)", fontweight="bold")
    ax.set_ylabel("Wilks Points")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val:.1f}",
                ha="center", fontweight="bold", fontsize=9)

    # Chart B — Feature importance
    ax = axes[1]
    bars = ax.barh(importance_df["Feature"][::-1],
                   importance_df["Importance"][::-1],
                   color="#55A868", edgecolor="white")
    ax.set_title("Random Forest\nFeature Importance", fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.spines[["top", "right"]].set_visible(False)
    for bar in bars:
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", fontsize=9)

    # Chart C — Actual vs Predicted — LightGBM (3000-point sample)
    ax = axes[2]
    s_idx = np.random.default_rng(42).choice(len(y_lgb_test),
                                              size=min(3000, len(y_lgb_test)),
                                              replace=False)
    ax.scatter(y_lgb_test[s_idx], lgb_preds[s_idx],
               alpha=0.3, s=8, color="#1A3A6B", label="LGB predictions")
    mn, mx = y_lgb_test.min(), y_lgb_test.max()
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Wilks Score")
    ax.set_ylabel("Predicted Wilks Score")
    ax.set_title(f"LGB Actual vs Predicted\nR² = {lgb_r2:.3f}", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("ml_evaluation.png", dpi=120)
    plt.close()
    print("Saved → ml_evaluation.png")

    # Chart D — Residuals distribution
    residuals = y_test - rf_preds
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=80, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axvline(0,       color="red",    linestyle="--", linewidth=1.5,
               label="Zero error")
    ax.axvline( rf_mae, color="orange", linestyle="--", linewidth=1.2,
               label=f"+MAE ({rf_mae:.1f})")
    ax.axvline(-rf_mae, color="orange", linestyle="--", linewidth=1.2,
               label=f"-MAE ({rf_mae:.1f})")
    ax.set_xlabel("Prediction Error (Actual − Predicted Wilks)")
    ax.set_ylabel("Number of Athletes")
    ax.set_title("Residuals Distribution — Random Forest", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("ml_residuals.png", dpi=120)
    plt.close()
    print("Saved → ml_residuals.png")


    # ── STAGE 5 — Save Trained Model ────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 5 — Saving trained model to disk")
    print("=" * 60)

    joblib.dump(rf,          MODEL_PATH)
    joblib.dump(encoders,    ENCODERS_PATH)
    joblib.dump(lr_pipeline, LR_PIPELINE_PATH)
    joblib.dump(lgb_model,   LGB_MODEL_PATH)
    joblib.dump(encoders,    LGB_ENCODERS_PATH)
    print(f"RF model saved     → {MODEL_PATH}")
    print(f"RF encoders saved  → {ENCODERS_PATH}")
    print(f"LR pipeline saved  → {LR_PIPELINE_PATH}")
    print(f"LGB model saved    → {LGB_MODEL_PATH}")
    print(f"LGB encoders saved → {LGB_ENCODERS_PATH}")


    # ── STAGE 6 — predict_wilks() Demo ──────────────────────
    print("\n" + "=" * 60)
    print("STAGE 6 — predict_wilks() function demo")
    print("=" * 60)

    # sex, equipment, age_group, bw, wc, raw_age, label
    # raw_age is passed to LGB (Age feature); year set to 2016 (dataset midpoint)
    demo_athletes = [
        ("M", "Raw",        "Open (24-39)",   83.0,  83.0, 30, "Male Open Raw 83kg"),
        ("F", "Raw",        "Junior (18-23)", 63.0,  63.0, 21, "Female Junior Raw 63kg"),
        ("M", "Single-ply", "Open (24-39)",  100.0, 100.0, 30, "Male Open Single-ply 100kg"),
        ("M", "Raw",        "Master 50-59",   90.0,  93.0, 53, "Male Master 50-59 Raw 90kg"),
        ("F", "Multi-ply",  "Open (24-39)",   72.0,  72.0, 28, "Female Open Multi-ply 72kg"),
    ]

    # LGB demo treats everyone as a first-time competitor (no history).
    # Age and Year are passed to give LGB its strongest non-history signals.
    print(f"\n{'Athlete Profile':<40} {'RF':>8} {'LR':>8} {'LGB':>8}")
    print("-" * 66)
    for sex, equip, age_grp, bw, wc, raw_age, label in demo_athletes:
        pred_rf  = predict_wilks(sex, equip, age_grp, bw, wc,
                                 model=rf, encoders=encoders)
        pred_lr  = predict_wilks_lr(sex, equip, age_grp, bw, wc,
                                    lr_pipeline=lr_pipeline)
        pred_lgb = predict_wilks_lgb(sex, equip, age_grp, bw, wc,
                                     age=raw_age, year=2016,
                                     model=lgb_model, encoders=encoders)
        print(f"{label:<40} {pred_rf:>8.1f} {pred_lr:>8.1f} {pred_lgb:>8.1f}")

    print("\n" + "=" * 60)
    print("ML MODEL COMPLETE")
    print("Charts : ml_wilks_distribution.png, ml_evaluation.png, ml_residuals.png")
    print(f"RF     : {MODEL_PATH}  |  Encoders : {ENCODERS_PATH}")
    print(f"LR     : {LR_PIPELINE_PATH}")
    print(f"LGB    : {LGB_MODEL_PATH}  |  Encoders : {LGB_ENCODERS_PATH}")
    print("=" * 60)