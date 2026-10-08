"""Tolerance-adjusted F1 (±1h) with neighbour lookup in the full ground truth.

Loads <input_label>_<site>_<sample_rate>.csv from the model folder and uses its
slick_tag=1 timestamps as the reference set for false-positive tolerance.

Also computes a baseline adjusted F1 from a simple windspeed rule (predict 1 if
wspd <= WSPD_THRESHOLD else 0); the windspeed column is auto-detected
(windspeed_336, windspeed_12, windspeed_3, ...) from the seed prediction CSVs.

Folder parsing, the wind baseline and adjusted F1 itself come from eval_common.py.
"""

from pathlib import Path
import argparse
import pandas as pd

from eval_common import (
    WSPD_THRESHOLD,
    adjusted_f1,
    load_full_gt_true_pos,
    parse_folder,
    seed_prediction_csvs,
    wind_baseline_pred,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("folder", nargs="+", type=Path)
args = p.parse_args()

# ---------------------------------------------------------------------------
# Adjusted F1 computation
# ---------------------------------------------------------------------------
def adjusted_f1_for_folder(folder: Path) -> dict:
    """Compute per-seed adjusted F1 (model and wind baseline) for a run folder, and
    aggregate to mean/std across seeds (std 0.0 if only one seed is present)."""
    input_label, site, sample_rate = parse_folder(folder.name)
    csvs = seed_prediction_csvs(folder, input_label, site, sample_rate)
    if not csvs:
        raise FileNotFoundError(f"No seed CSVs in {folder}")
    full_true_pos = load_full_gt_true_pos(folder, input_label, site, sample_rate)

    per_seed = []
    baseline_per_seed = []
    for c in csvs:
        df = pd.read_csv(c)
        y_true = df["y_true"].to_numpy(int)
        per_seed.append(adjusted_f1(
            df["Time"], y_true, df["y_pred"].to_numpy(int), full_true_pos,
        ))
        # baseline: wspd <= threshold -> predict 1
        baseline_per_seed.append(adjusted_f1(
            df["Time"], y_true, wind_baseline_pred(df), full_true_pos,
        ))

    d = pd.DataFrame(per_seed)
    b = pd.DataFrame(baseline_per_seed)
    return {
        "mean": d.mean().to_dict(),
        "std":  (d.std(ddof=1).to_dict() if len(per_seed) > 1
                 else {col: 0.0 for col in d.columns}),
        "baseline_mean": b.mean().to_dict(),
        "baseline_std": (b.std(ddof=1).to_dict() if len(baseline_per_seed) > 1
                         else {col: 0.0 for col in b.columns}),
        "n_seeds": len(per_seed),
        "n_full_true_pos": int(len(full_true_pos)),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for folder in args.folder:
        r = adjusted_f1_for_folder(folder)
        print(f"Folder: {folder.name}")
        print(f"Adjusted F1 (±1h, full-GT neighbour lookup) over {r['n_seeds']} seeds")
        print(f"  full-GT positives: {r['n_full_true_pos']}")
        for k in ("adj_f1", "adj_precision", "adj_recall"):
            print(f"  {k:14}: {r['mean'][k]:.3f} ± {r['std'][k]:.3f}")
        print(f"Baseline adjusted F1 (wspd <= {WSPD_THRESHOLD:g} m/s -> slick)")
        for k in ("adj_f1", "adj_precision", "adj_recall"):
            print(f"  {k:14}: {r['baseline_mean'][k]:.3f} ± {r['baseline_std'][k]:.3f}")