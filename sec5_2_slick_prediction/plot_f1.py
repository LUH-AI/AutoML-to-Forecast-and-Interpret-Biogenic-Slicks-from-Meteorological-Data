"""F1 and adjusted F1 against input length: the single-site figure (Figure 5) and the combined-
dataset figure (Figure 6), both built from exactly the same numbers.

For every model folder under the results directory it computes three series:

    - F1 of the AutoGluon model, mean and std over seeds
    - Adjusted F1 (a prediction counts as correct when a real slick falls within the
      tolerance window), mean and std over seeds
    - F1 of the wind-speed baseline (slick whenever the most recent reading is <= 5
      m/s), a fixed rule, so no spread

The adjusted score looks neighbours up in the *full* ground truth, not just the test
split, so a prediction next to a slick that ended up in the training split still
counts as a near miss rather than a false positive. It writes the tidy table behind every
point (metrics_f1_adjf1.csv) next to the figures.

Folder parsing, the wind baseline and adjusted F1 itself come from eval_common.py.

Run from the project root:
    python sec5_2_slick_prediction/plot_f1.py                  # both figures
    python sec5_2_slick_prediction/plot_f1.py --figure combined
    python sec5_2_slick_prediction/plot_f1.py -d results/no_static -o results/figures --style markers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from eval_common import (
    WSPD_THRESHOLD,
    adjusted_f1,
    find_wspd_col,
    label_to_display,
    label_to_minutes,
    load_full_gt_true_pos,
    parse_folder,
    seed_prediction_csvs,
    wind_baseline_pred,
)

# Site order and titles for the single-site figure.
SINGLE_SITES = [
    ("fino1_2022", "Fino1 (2022)"),
    ("spiekeroog_2021", "Spiekeroog (2021)"),
    ("spiekeroog_2024", "Spiekeroog (2024)"),
]

DEFAULT_RESULTS_DIR = "results/no_static"
DEFAULT_OUTPUT_DIR = "results/figures"

PUB_STYLE = {
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":        11,
    "axes.labelsize":   11,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "axes.linewidth":   0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.linewidth":   0.6,
    "grid.alpha":       0.25,
    "savefig.dpi":      300,
}

# Okabe-Ito: one blue family for the two AutoGluon metrics, vermillion for the
# baseline reference. Safe for all common colour-vision deficiencies.
COL_F1       = "#0072B2"   # dark blue
COL_ADJ      = "#56B4E9"   # sky blue
COL_BASELINE = "#D55E00"   # vermillion


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the command-line arguments."""
    parser.add_argument("-d", "--data_path", type=str, default=DEFAULT_RESULTS_DIR,
                        help=f"Folder holding one subfolder per model (default: {DEFAULT_RESULTS_DIR})")
    parser.add_argument("-o", "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Where to write the figure and the metrics CSV (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--tolerance", type=str, default="1h",
                        help="Adjusted-F1 tolerance, parseable by pd.Timedelta (default: 1h)")
    parser.add_argument("--style", choices=["bars", "markers"], default="bars",
                        help="bars is the published style; markers shows each score as a "
                             "point with its error bar (default: bars)")
    parser.add_argument("--x_scale", choices=["categorical", "log"], default="categorical",
                        help="Spacing of the input-length axis for --style markers. "
                             "categorical spaces labels evenly, log places each model at "
                             "its true input length (default: categorical)")


# ---------------------------------------------------------------------------
# Collecting the metrics into one tidy table
# ---------------------------------------------------------------------------
def _agg(values: list[float]) -> tuple[float, float]:
    """Return (mean, std) across seeds; std is 0.0 for a single value."""
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std


def evaluate_folder(folder: Path, tolerance: pd.Timedelta) -> list[dict] | None:
    """AutoGluon F1 and adjusted F1 for one model folder, mean and std over seeds."""
    input_label, site, sample_rate = parse_folder(folder.name)
    pred_csvs = seed_prediction_csvs(folder, input_label, site, sample_rate)
    if not pred_csvs:
        print(f"  skip (no seed predictions): {folder.name}")
        return None
    full_true_pos = load_full_gt_true_pos(folder, input_label, site, sample_rate)

    f1_vals, adj_vals = [], []
    for csv in pred_csvs:
        df = pd.read_csv(csv)
        y_true = df["y_true"].to_numpy(int)
        y_pred = df["y_pred"].to_numpy(int)
        f1_vals.append(f1_score(y_true, y_pred))
        adj_vals.append(
            adjusted_f1(df["Time"], y_true, y_pred, full_true_pos, tolerance)["adj_f1"])

    base_row = {
        "site": site,
        "input_label": input_label,
        "input_minutes": label_to_minutes(input_label),
        "model": "autogluon",
        "n_seeds": len(pred_csvs),
    }
    rows = []
    for metric, vals in [("f1", f1_vals), ("adj_f1", adj_vals)]:
        mean, std = _agg(vals)
        rows.append({**base_row, "metric": metric, "point": mean, "std": std})
    return rows


def evaluate_baseline_folder(folder: Path) -> list[dict] | None:
    """F1 of the wind-speed rule: slick whenever the most recent reading is <= 5 m/s."""
    input_label, site, sample_rate = parse_folder(folder.name)
    pred_csvs = seed_prediction_csvs(folder, input_label, site, sample_rate)
    if not pred_csvs:
        return None
    df = pd.read_csv(pred_csvs[0])
    if find_wspd_col(df, required=False) is None:
        print(f"  no windspeed_* column in {pred_csvs[0].name}, skipping baseline")
        return None
    y_true = df["y_true"].to_numpy(int)
    y_pred = wind_baseline_pred(df)
    return [{
        "site": site,
        "input_label": input_label,
        "input_minutes": label_to_minutes(input_label),
        "model": "baseline",
        "n_seeds": 1,
        "metric": "f1",
        "point": float(f1_score(y_true, y_pred)),
        "std": 0.0,
    }]


def collect_metrics(results_dir: Path, tolerance: pd.Timedelta) -> pd.DataFrame:
    """Walk every model folder and return one tidy table of all three series."""
    folders = sorted(p for p in results_dir.iterdir() if p.is_dir())
    print(f"Found {len(folders)} directories under {results_dir}")

    rows: list[dict] = []
    for folder in folders:
        print(f"  evaluating: {folder.name}")
        try:
            r = evaluate_folder(folder, tolerance)
            if r:
                rows.extend(r)
            b = evaluate_baseline_folder(folder)
            if b:
                rows.extend(b)
        except (ValueError, FileNotFoundError) as e:
            print(f"  skip {folder.name}: {e}")
            continue

    if not rows:
        raise SystemExit(f"No prediction files found under {results_dir}. Nothing to plot.")
    return pd.DataFrame(rows)


def write_metrics_csv(all_results: pd.DataFrame, output_dir: Path) -> Path:
    """Write the tidy table next to the figures, so the numbers can be checked."""
    path = output_dir / "metrics_f1_adjf1.csv"
    all_results.to_csv(path, index=False)
    print(f"Wrote metrics CSV: {path}")
    return path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_bars(ax, df_site: pd.DataFrame, tolerance_label: str,
              show_legend: bool = False) -> None:
    """Grouped bars for one site: F1 and adjusted F1, with the baseline as a line.

    The baseline is one fixed rule rather than a per-input-length model, so it is drawn
    as a single horizontal reference line instead of a third bar.
    """
    df_unique = df_site.drop_duplicates("input_label").sort_values("input_minutes")
    labels = df_unique["input_label"].tolist()
    x = np.arange(len(labels))
    width = 0.38

    f1_df = df_site[(df_site.model == "autogluon") & (df_site.metric == "f1")].sort_values("input_minutes")
    adj_df = df_site[(df_site.model == "autogluon") & (df_site.metric == "adj_f1")].sort_values("input_minutes")

    err_kw = dict(elinewidth=0.8, capthick=0.8)
    ax.bar(x - width / 2, f1_df["point"].to_numpy(), width, yerr=f1_df["std"].to_numpy(),
           color=COL_F1, edgecolor=COL_F1, linewidth=0.4,
           capsize=2.5, error_kw=err_kw, label="AutoGluon F1", zorder=2)
    ax.bar(x + width / 2, adj_df["point"].to_numpy(), width, yerr=adj_df["std"].to_numpy(),
           color=COL_ADJ, edgecolor=COL_F1, linewidth=0.4,
           capsize=2.5, error_kw=err_kw, label=f"Adj. F1 (±{tolerance_label} tol.)", zorder=2)

    baseline_df = df_site[(df_site.model == "baseline") & (df_site.metric == "f1")]
    if not baseline_df.empty:
        ax.axhline(float(baseline_df["point"].mean()),
                   color=COL_BASELINE, linestyle=(0, (4, 2)),
                   linewidth=1.5, zorder=3, label=f"Wind ≤ {WSPD_THRESHOLD:g} m/s F1")

    ax.set_xticks(x)
    ax.set_xticklabels([label_to_display(l) for l in labels],
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xlabel("Input length")
    ax.set_ylabel("F1 score")
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0.0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        ax.legend(loc="upper left", framealpha=0.95, edgecolor="none",
                  borderpad=0.4, handlelength=1.6, handletextpad=0.5).set_zorder(6)


def plot_markers(ax, df_site: pd.DataFrame, tolerance_label: str,
                 show_legend: bool = False, x_scale: str = "categorical") -> None:
    """One point per score with its error bar.

    AutoGluon markers are deliberately not joined: each input length is a separately
    trained model, so a line would imply an interpolation that doesn't exist. The
    baseline is one rule applied to every test set, so joining it is fine.
    """
    series = [
        {"key": ("autogluon", "f1"), "colour": "tab:blue", "marker": "o",
         "label": "AutoGluon F1", "linestyle": "none", "filled": True},
        {"key": ("autogluon", "adj_f1"), "colour": "tab:green", "marker": "^",
         "label": f"AutoGluon Adj. F1 (±{tolerance_label})", "linestyle": "none", "filled": True},
        {"key": ("baseline", "f1"), "colour": "#555555", "marker": "s",
         "label": f"Wind ≤ {WSPD_THRESHOLD:g} m/s F1 (baseline)", "linestyle": "--",
         "filled": False},
    ]

    df_unique = df_site.drop_duplicates("input_label").sort_values("input_minutes")
    labels = df_unique["input_label"].tolist()
    minutes = df_unique["input_minutes"].tolist()

    if x_scale == "log":
        pos = dict(zip(labels, minutes))
        x_ticks = minutes
    else:
        pos = {lab: i for i, lab in enumerate(labels)}
        x_ticks = list(range(len(labels)))

    for s in series:
        model, metric = s["key"]
        sub = df_site[(df_site["model"] == model) & (df_site["metric"] == metric)]
        if sub.empty:
            continue
        sub = sub.sort_values("input_minutes")
        ax.errorbar(
            np.array([pos[l] for l in sub["input_label"]], dtype=float),
            sub["point"].to_numpy(), yerr=sub["std"].to_numpy(),
            fmt=s["marker"], linestyle=s["linestyle"],
            color=s["colour"], ecolor=s["colour"], markersize=8,
            markerfacecolor=(s["colour"] if s["filled"] else "white"),
            markeredgecolor=("black" if s["filled"] else s["colour"]),
            markeredgewidth=(0.6 if s["filled"] else 1.2),
            linewidth=1.2, capsize=3, elinewidth=1.2,
            label=s["label"], zorder=3 if s["filled"] else 2,
        )

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([label_to_display(l) for l in labels], rotation=30, ha="right")
    if x_scale == "log":
        ax.set_xscale("log")
        ax.minorticks_off()          # set_xscale re-installs minor ticks; they clutter
        ax.set_xlim(minutes[0] / 1.3, minutes[-1] * 1.3)
    else:
        ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_xlabel("Input length")
    ax.set_ylabel("Score")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0.2, 1.0)
    if show_legend:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)


def prepare(args) -> tuple[pd.DataFrame, Path, pd.Timedelta]:
    """Resolve the paths, collect the metrics and write the CSV."""
    results_dir = Path(args.data_path)
    output_dir = Path(args.output_dir)
    if not results_dir.is_dir():
        raise SystemExit(
            f"{results_dir} is not a directory. Run this from the project root, or pass "
            f"-d with the folder that holds one subfolder per model."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    tolerance = pd.Timedelta(args.tolerance)

    all_results = collect_metrics(results_dir, tolerance)
    write_metrics_csv(all_results, output_dir)
    return all_results, output_dir, tolerance


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_single_site(all_results, out_path, tolerance_label, style, x_scale) -> None:
    """Draw the three single-site panels with a shared y-axis."""
    with plt.rc_context(PUB_STYLE if style == "bars" else {}):
        figsize = (11.0, 3.6) if style == "bars" else (len(SINGLE_SITES) * 5.5, 4.5)
        fig, axes = plt.subplots(1, len(SINGLE_SITES), figsize=figsize, sharey=True)
        axes = np.atleast_1d(axes)
        panel_letters = "abc"

        for i, (ax, (site, title)) in enumerate(zip(axes, SINGLE_SITES)):
            sub = all_results[all_results.site == site]
            if sub.empty:
                print(f"  no results for {site}, leaving that panel empty")
                ax.set_visible(False)
                continue
            if style == "bars":
                # No per-panel legend: one figure-level legend above the panels cannot
                # cover a bar in any of them.
                plot_bars(ax, sub, tolerance_label, show_legend=False)
                ax.set_title(f"({panel_letters[i]}) {title}", loc="left", fontweight="bold")
            else:
                plot_markers(ax, sub, tolerance_label, show_legend=(i == 0), x_scale=x_scale)
                ax.set_title(title)
            if i > 0:
                ax.set_ylabel("")

        if style == "bars":
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                           ncol=len(handles), frameon=False, fontsize=9,
                           handletextpad=0.6, columnspacing=2.0)
            plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
        else:
            plt.tight_layout()

        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote {out_path}")


def plot_combined(all_results, out_path, tolerance_label, style, x_scale) -> None:
    """Draw the single panel of the model trained on all three sites at once."""
    sub = all_results[all_results.site == "combined"]
    if sub.empty:
        raise SystemExit(
            "No results for the 'combined' model. Train it first, or check that a "
            "*_combined_* folder exists under the results directory."
        )

    with plt.rc_context(PUB_STYLE if style == "bars" else {}):
        fig, ax = plt.subplots(figsize=(8, 3.5) if style == "bars" else (7.0, 3.6))
        if style == "bars":
            plot_bars(ax, sub, tolerance_label, show_legend=True)
            ax.set_title("Combined", loc="left", fontweight="bold")
        else:
            plot_markers(ax, sub, tolerance_label, show_legend=True, x_scale=x_scale)
            ax.set_title("Combined")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--figure", choices=["single_site", "combined", "both"], default="both",
                        help="Which figure to draw (default: both)")
    args = parser.parse_args()

    all_results, output_dir, _ = prepare(args)
    suffix = "" if args.style == "bars" else f"_{args.style}"
    if args.style == "markers" and args.x_scale != "categorical":
        suffix += f"_{args.x_scale}"
    if args.figure in ("single_site", "both"):
        plot_single_site(all_results, output_dir / f"f1_single_site{suffix}.png",
                         args.tolerance, args.style, args.x_scale)
    if args.figure in ("combined", "both"):
        plot_combined(all_results, output_dir / f"f1_combined{suffix}.png",
                      args.tolerance, args.style, args.x_scale)


if __name__ == "__main__":
    main()
