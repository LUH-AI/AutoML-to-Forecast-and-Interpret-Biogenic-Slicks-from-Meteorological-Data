"""Precision-Recall curves per dataset, one line per input-length model.

For each site, every input-length model contributes one PR curve, computed by
interpolating each seed's curve onto a fixed recall grid and averaging across
seeds; a thin shaded band shows ±std across seeds.

Reference markers per panel:
    - Dashed gray horizontal line at the positive-class prevalence
      (a random classifier's PR curve sits at this level — "no skill").
    - Black star at the wind ≤ 5 m/s rule's (recall, precision) operating point.

Outputs two PNG files:
    pr_curves_single_site.png    three single-site panels (shared y)
    pr_curves_combined.png       combined-site panel

Folder parsing and the wind baseline come from eval_common.py.

Run from the project root:
    python sec5_2_slick_prediction/plot_pr_curves.py -d <results_dir> -o <output_dir>
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import precision_recall_curve

from eval_common import (
    find_wspd_col,
    label_to_minutes,
    parse_folder,
    seed_prediction_csvs,
    wind_baseline_pred,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-d", "--data_path",  type=str, required=True,
                    help="Folder containing per-model subfolders.")
parser.add_argument("-o", "--output_dir", type=str, required=True,
                    help="Where to save the figures.")
parser.add_argument("--no_band", action="store_true",
                    help="Disable the ±std shaded band around each PR curve.")
parser.add_argument("--no_baseline_point", action="store_true",
                    help="Disable the wind ≤ 5 m/s baseline operating-point marker.")
args = parser.parse_args()

RESULTS_DIR = Path(args.data_path)
OUTPUT_DIR  = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Common grid for interpolating per-seed PR curves before averaging.
RECALL_GRID = np.linspace(0.0, 1.0, 201)


plt.rcParams.update({
    "font.size":       11,
    "axes.titlesize":  13,
    "axes.labelsize":  11,
    "legend.fontsize":  8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})



# ---------------------------------------------------------------------------
# PR curve computation
# ---------------------------------------------------------------------------
def _interp_pr(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Precision interpolated onto RECALL_GRID."""
    prec, rec, _ = precision_recall_curve(y_true, proba)
    # `precision_recall_curve` returns recall from high threshold (=0) to low
    # threshold (=1), so recall is non-increasing here — sort ascending for interp.
    order      = np.argsort(rec)
    rec_sorted = rec[order]
    prec_sorted = prec[order]
    return np.interp(RECALL_GRID, rec_sorted, prec_sorted,
                     left=prec_sorted[0], right=prec_sorted[-1])


def compute_model_curves(folder: Path) -> dict | None:
    """Mean PR curve across seeds for one input-length model."""
    input_label, site, sample_rate = parse_folder(folder.name)
    pred_csvs = seed_prediction_csvs(folder, input_label, site, sample_rate)
    if not pred_csvs:
        return None

    interp_precs = []
    prevalences = []
    for csv in pred_csvs:
        df     = pd.read_csv(csv)
        y_true = df["y_true"].to_numpy(int)
        proba  = df["proba_1"].to_numpy(float)
        interp_precs.append(_interp_pr(y_true, proba))
        prevalences.append(float(y_true.mean()))

    p_arr = np.asarray(interp_precs)
    return {
        "site":          site,
        "input_label":   input_label,
        "input_minutes": label_to_minutes(input_label),
        "n_seeds":       len(pred_csvs),
        "prevalence":    float(np.mean(prevalences)),
        "recall":        RECALL_GRID,
        "prec_mean":     p_arr.mean(axis=0),
        "prec_std":      (p_arr.std(axis=0, ddof=1) if len(interp_precs) > 1
                          else np.zeros_like(p_arr[0])),
    }


def compute_baseline_point(folder: Path) -> dict | None:
    """Single (precision, recall) for the wind ≤ 5 m/s rule.

    Computed on the first seed CSV — the rule is deterministic, so any seed's
    test slice gives the same operating point (modulo seed-dependent splits).
    """
    input_label, site, sample_rate = parse_folder(folder.name)
    pred_csvs = seed_prediction_csvs(folder, input_label, site, sample_rate)
    if not pred_csvs:
        return None
    df = pd.read_csv(pred_csvs[0])
    if find_wspd_col(df, required=False) is None:
        return None
    y_true = df["y_true"].to_numpy(int)
    y_pred = wind_baseline_pred(df)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    return {"site": site, "input_label": input_label,
            "precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _colour_for(input_minutes: int, all_minutes: list[int]):
    """Map an input length to a viridis colour, log-spaced over the range."""
    if len(set(all_minutes)) <= 1:
        return cm.viridis(0.5)
    log_m   = np.log(input_minutes)
    log_min = np.log(min(all_minutes))
    log_max = np.log(max(all_minutes))
    t = (log_m - log_min) / (log_max - log_min)
    # Pull off the very bright end of viridis — yellow on white is hard to see.
    return cm.viridis(0.05 + 0.85 * t)

def pretty_label(label: str) -> str:
    """Expand a compact input label for display, e.g. '2h' -> '2 hours'."""
    if label.endswith("h"):
        n = int(label.removesuffix("h"))
        if n % 24 == 0:                      # whole days read better as days
            d = n // 24
            return "1 day" if d == 1 else f"{d} days"
        return f"{n} hour" if n == 1 else f"{n} hours"
    if label.endswith("day") or label.endswith("days"):
        n = label.removesuffix("days").removesuffix("day")
        return f"{n} day" if n == "1" else f"{n} days"
    if label.endswith("week") or label.endswith("weeks"):
        n = label.removesuffix("weeks").removesuffix("week")
        return f"{n} week" if n == "1" else f"{n} weeks"
    return label


def _plot_one_pr_panel(ax, curves: list[dict], baselines: list[dict],
                       title: str | None = None, ylabel: bool = True) -> None:
    """Draw one site's PR curves (one per input-length model) plus the no-skill
    line and wind-baseline star onto ax."""
    if not curves:
        ax.set_visible(False)
        return
    curves = sorted(curves, key=lambda c: c["input_minutes"])
    all_minutes = [c["input_minutes"] for c in curves]

    for c in curves:
        colour = _colour_for(c["input_minutes"], all_minutes)
        lbl = pretty_label(c["input_label"])
        ax.plot(c["recall"], c["prec_mean"],
                color=colour, linewidth=1.6, label=lbl, zorder=3)
        if not args.no_band and np.any(c["prec_std"] > 0):
            ax.fill_between(c["recall"],
                            np.clip(c["prec_mean"] - c["prec_std"], 0.0, 1.0),
                            np.clip(c["prec_mean"] + c["prec_std"], 0.0, 1.0),
                            color=colour, alpha=0.12, linewidth=0, zorder=2)

    # No-skill horizontal at prevalence (average across this site's models)
    prevalence = float(np.mean([c["prevalence"] for c in curves]))
    ax.axhline(prevalence, linestyle="--", color="gray", linewidth=1.0,
                label="no-skill",
                zorder=1)

    # Wind-baseline operating point — single black star
    if baselines and not args.no_baseline_point:
        prec = float(np.mean([b["precision"] for b in baselines]))
        rec  = float(np.mean([b["recall"]    for b in baselines]))
        ax.scatter([rec], [prec], marker="*", s=140,
                   color="black", edgecolor="white", linewidth=0.8,
                   label="wind ≤ 5 m/s",
                   zorder=5)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Recall")
    if ylabel:
        ax.set_ylabel("Precision")
    if title:
        ax.set_title(title, pad=8)
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="lower left", framealpha=0.92,
        labelspacing=0.3,    # vertical gap between rows
        handlelength=1.6,    # length of the color line sample
        handletextpad=0.5,   # gap between sample and text
        borderpad=0.4,       # padding inside the frame
        markerscale=0.8,     # shrinks the s=140 star in the legend only
        ncol=2
    )


def plot_pr_single_site(curves_by_site: dict[str, list[dict]],
                        baselines_by_site: dict[str, list[dict]],
                        out_path: Path) -> None:
    """Draw the three-panel (one per single-site dataset) PR figure and save it."""
    sites = [
        ("fino1_2022",  "Fino1 (2022)"),
        ("spiekeroog_2021", "Spiekeroog (2021)"),
        ("spiekeroog_2024", "Spiekeroog (2024)"),
    ]
    fig, axes = plt.subplots(1, len(sites),
                             figsize=(len(sites) * 6, 5),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for i, (ax, (site, title)) in enumerate(zip(axes, sites)):
        _plot_one_pr_panel(ax,
                           curves_by_site.get(site, []),
                           baselines_by_site.get(site, []),
                           title=title,
                           ylabel=(i == 0))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_pr_combined(curves_by_site: dict[str, list[dict]],
                     baselines_by_site: dict[str, list[dict]],
                     out_path: Path) -> None:
    """Draw the single-panel combined-dataset PR figure and save it."""
    if "combined" not in curves_by_site:
        print("  no combined-site curves, skipping combined PR figure")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _plot_one_pr_panel(ax,
                       curves_by_site["combined"],
                       baselines_by_site.get("combined", []),
                       ylabel=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Compute every model's PR curve and wind-baseline point, then render both figures."""
    folders = sorted(p for p in RESULTS_DIR.iterdir() if p.is_dir())
    print(f"Found {len(folders)} directories under {RESULTS_DIR}")

    curves_by_site: dict[str, list[dict]] = defaultdict(list)
    baselines_by_site: dict[str, list[dict]] = defaultdict(list)
    for folder in folders:
        print(f"  evaluating: {folder.name}")
        try:
            c = compute_model_curves(folder)
            if c is not None:
                curves_by_site[c["site"]].append(c)
            b = compute_baseline_point(folder)
            if b is not None:
                baselines_by_site[b["site"]].append(b)
        except (ValueError, FileNotFoundError) as e:
            print(f"  skip {folder.name}: {e}")
            continue

    if not curves_by_site:
        print("No prediction files found. Nothing to plot.")
        return

    plot_pr_single_site(curves_by_site, baselines_by_site,
                        OUTPUT_DIR / "pr_curves_single_site.png")
    plot_pr_combined(curves_by_site, baselines_by_site,
                     OUTPUT_DIR / "pr_curves_combined.png")

if __name__ == "__main__":
    main()