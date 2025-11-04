import numpy as np
import pandas as pd
from pathlib import Path

# === 1. path configuration ===
OLD_PATH = Path("data/mushrooms.csv")
NEW_PATH = Path("data/mushrooms_enhanced.csv")
OUTPUT_PATH = Path("data/output/mushrooms_new_imputed.csv")

# === 2. read and load data ===
print("🔹 Loading datasets...")
df_old = pd.read_csv(OLD_PATH)
df_new = pd.read_csv(NEW_PATH)

print(f"✅ Old dataset shape: {df_old.shape}")
print(f"✅ New dataset shape: {df_new.shape}")

# === 3. merge datasets ===
print("🔹 Combining old and new datasets...")
df_combined = pd.concat([df_old, df_new], ignore_index=True)

# === 4. remove the 'stalk-root' feature ===
if "stalk-root" in df_combined.columns:
    print("🔹 Removing feature: 'stalk-root'")
    df_combined = df_combined.drop(columns=["stalk-root"])
else:
    print("⚠️ Feature 'stalk-root' not found — skipping.")

# === 5. identify features (excluding label) ===
all_features = [col for col in df_combined.columns if col != "class"]
print(f"✅ Will impute {len(all_features)} features (excluding label 'class').\n")

# === 6. replace '?' with NaN ===
print("🔹 Replacing '?' with NaN...")
df_combined = df_combined.replace('?', np.nan)

# === 7. remove rows with >=3 missing values ===
print("🔹 Removing rows with 3 or more missing values before imputation...")
initial_rows = len(df_combined)
df_combined = df_combined[df_combined[all_features].isna().sum(axis=1) < 3].reset_index(drop=True)
removed = initial_rows - len(df_combined)
percent_removed = removed / initial_rows * 100
print(f"  • Removed {removed} rows ({percent_removed:.1f}% of total, kept {len(df_combined)} rows).\n")

# === 8. define helper functions ===
def safe_baseline_fill(df, features, missing_token="missing"):
    """Replace NaN with a safe token to avoid processing errors."""
    dfb = df.copy()
    for col in features:
        if col in dfb.columns:
            dfb[col] = dfb[col].astype("object").where(dfb[col].notna(), missing_token)
    return dfb


def conditional_mc_impute(
    df_new, df_ref, features,
    random_state=42, min_subset=15, alpha=1.0, tau=50
):
    """Conditional Monte Carlo imputation for categorical data."""
    rng = np.random.default_rng(random_state)
    df_filled = df_new.copy()

    # === category options per column ===
    category_choices = {
        col: sorted(df_ref[col].dropna().astype("object").unique().tolist())
        for col in features if col in df_ref.columns
    }

    # === global probability distributions ===
    global_counts = {col: df_ref[col].value_counts() for col in features if col in df_ref.columns}
    global_probs = {}
    for col in category_choices:
        cats = category_choices[col]
        cnt = global_counts[col].reindex(cats).fillna(0.0).to_numpy(dtype=float)
        K = len(cats)
        p = (cnt + alpha) / (cnt.sum() + alpha * K)
        global_probs[col] = p

    # === imputation loop ===
    for col in features:
        if col not in df_filled.columns:
            continue
        miss_idx = df_filled.index[df_filled[col].isna()]
        if len(miss_idx) == 0:
            continue

        cats = category_choices[col]
        gprob = global_probs[col]

        print(f"  • Imputing {len(miss_idx)} missing values in '{col}'...")

        for idx in miss_idx:
            known = {
                k: df_filled.at[idx, k]
                for k in features
                if k in df_filled.columns and k != col and pd.notna(df_filled.at[idx, k])
            }

            subset = df_ref
            for k, v in known.items():
                if k in subset.columns:
                    subset = subset[subset[k] == v]

            if len(subset) > 0:
                cnt = subset[col].value_counts().reindex(cats).fillna(0.0).to_numpy(dtype=float)
            else:
                cnt = np.zeros(len(cats), dtype=float)

            K = len(cats)
            p_subset = (cnt + alpha) / (cnt.sum() + alpha * K)
            lam = tau / (tau + max(len(subset), 0.0))
            p_final = (1 - lam) * p_subset + lam * gprob
            p_final = p_final / p_final.sum()

            df_filled.at[idx, col] = rng.choice(cats, p=p_final)

    return df_filled


# === 9. run imputation on combined dataset ===
print("🔹 Starting baseline fill...")
df_combined_b = safe_baseline_fill(df_combined, all_features)

print("🔹 Running conditional Monte Carlo imputation on combined dataset...")
df_combined_imputed = conditional_mc_impute(
    df_combined, df_combined_b, all_features,
    random_state=42, min_subset=15, alpha=1.0, tau=50
)

# === 10. finalize & save ===
print("🔹 Converting any remaining '?' to 'missing'...")
df_combined_imputed = df_combined_imputed.replace('?', 'missing')

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_combined_imputed.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Saved imputed combined dataset to: {OUTPUT_PATH}")
print("Remaining missing values:", df_combined_imputed.isnull().sum().sum())