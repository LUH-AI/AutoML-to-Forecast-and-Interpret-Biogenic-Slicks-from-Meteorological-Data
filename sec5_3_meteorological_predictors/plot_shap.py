"""
Concatenate the per-job SHAP chunks written by calc_shap.py and render the plots.
"""
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-o", "--output_dir", type=str, required=True, help="Directory holding shap_values_job_*.npy and X_explain_job_*.csv")
parser.add_argument("--max_display", type=int, default=20, help="Number of features to show in the summary plot.")
args = parser.parse_args()

OUT_DIR = Path(args.output_dir)
if not OUT_DIR.is_dir():
    raise FileNotFoundError(f"Output dir not found: {OUT_DIR}")

# --------------------------------------------------------------------------- #
# Load and concatenate SHAP chunks
# --------------------------------------------------------------------------- #
val_files = sorted(OUT_DIR.glob("shap_values_job_*.npy"))
csv_files = sorted(OUT_DIR.glob("X_explain_job_*.csv"))

if not val_files:
    # Fall back to single-job layout
    single_val = OUT_DIR / "shap_values.npy"
    single_csv = OUT_DIR / "X_explain.csv"
    if not single_val.is_file():
        raise FileNotFoundError(f"No shap_values chunks in {OUT_DIR}")
    val_files = [single_val]
    csv_files = [single_csv]

print(f"Found {len(val_files)} chunk(s)")

job_ids_val = [int(re.search(r"job_(\d+)", p.name).group(1)) if "job_" in p.name else 0 for p in val_files]
job_ids_csv = [int(re.search(r"job_(\d+)", p.name).group(1)) if "job_" in p.name else 0 for p in csv_files]
assert job_ids_val == job_ids_csv, "shap_values_* and X_explain_* job indices do not match"

shap_chunks = [np.load(p) for p in val_files]
df_chunks = [pd.read_csv(p) for p in csv_files]

shap_values = np.concatenate(shap_chunks, axis=0)
X_explain = pd.concat(df_chunks, axis=0, ignore_index=True)

# Strip metadata columns added by calc_shap.py before plotting
meta_cols = [c for c in ("sample_id", "original_index", "y_true") if c in X_explain.columns]
X_features = X_explain.drop(columns=meta_cols)

print(f"shap_values shape: {shap_values.shape}")
print(f"X_explain shape:   {X_features.shape}")
assert shap_values.shape == X_features.shape, (
    f"Mismatch between SHAP values {shap_values.shape} and features {X_features.shape}"
)

feature_names = X_features.columns.tolist()

# Persist concatenated arrays for downstream plots
np.save(OUT_DIR / "shap_values_all.npy", shap_values)
X_explain.to_csv(OUT_DIR / "X_explain_all.csv", index=False)

# --------------------------------------------------------------------------- #
# Violin plot
# --------------------------------------------------------------------------- #
plt.figure()
shap.summary_plot(
    shap_values,
    X_features,
    feature_names=feature_names,
    max_display=args.max_display,
    plot_type="violin",
    show=False,
)
plt.tight_layout()
out_path = OUT_DIR / "shap_violin.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")


# --------------------------------------------------------------------------- #
# Dependence plots: most recent windspeed / temperature / humidity
# --------------------------------------------------------------------------- #
# Assumption: trailing _12 is the most recent timestep
DEPENDENCE_FEATURES = {
    "windspeed":   "windspeed_12",
    "temperature": "airtemperature_12",
    "humidity":    "humidity_12",
}

for label, feat in DEPENDENCE_FEATURES.items():
    if feat not in feature_names:
        print(f"[skip] feature '{feat}' not in dataset")
        continue
    print(f"Dependence plot for '{label}' -> {feat}")
    plt.figure()
    shap.dependence_plot(
        feat,
        shap_values,
        X_features,
        feature_names=feature_names,
        interaction_index=None,
        show=False,
    )
    plt.tight_layout()
    dep_path = OUT_DIR / f"shap_dependence_{label}.png"
    plt.savefig(dep_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {dep_path}")


# --------------------------------------------------------------------------- #
# Dependence plot: windspeed history
# --------------------------------------------------------------------------- #
FEATURES = [
    "windspeed_10","windspeed_11","windspeed_12"
]

fig, axs = plt.subplots(1,3,sharey=True, figsize=(16,5))

for feat, ax in zip(FEATURES, axs):
    if feat not in feature_names:
        raise KeyError(f"[ERR] feature '{feat}' not in dataset")
    shap.dependence_plot(
        feat,
        shap_values,
        X_features,
        feature_names=feature_names,
        interaction_index=None,
        show=False,
        ax=ax
    )

plt.tight_layout()
dep_path = OUT_DIR / "shap_wspd_history.png"
plt.savefig(dep_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved {dep_path}")
