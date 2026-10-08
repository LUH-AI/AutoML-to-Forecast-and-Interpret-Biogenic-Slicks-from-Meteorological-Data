r"""Describes the labelled datasets: how many slicks there are, and when.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# Labelled per-site tables at the sensor's own sampling rate (10 min, 1 h, 5 min), as
# written by `preprocess.py merge` into the site's own folder. Keyed by the label
# used in the figures.
NATIVE_FILES = {
    "Fino1 (2022)": "fino1_2022/sensors_with_labels.csv",
    "Spiekeroog (2021)": "spiekeroog_2021/sensors_with_labels.csv",
    "Spiekeroog (2024)": "spiekeroog_2024/sensors_with_labels.csv",
}
COMBINED_FILE = "1hour_combined_dataset.csv"

# `site` values in the combined dataset -> the label used in the figures.
SITE_LABELS = {
    "fino1_2022": "Fino1 (2022)",
    "spiekeroog_2021": "Spiekeroog (2021)",
    "spiekeroog_2024": "Spiekeroog (2024)",
}

COL_TRUE = "#1f77b4"
COL_FALSE = "#ff7f0e"
DPI = 200

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("-d", "--dataset_dir", type=str, default="build",
                    help="Build folder holding the hourly datasets and the per-site "
                         "folders (default: build).")
parser.add_argument("-c", "--combined", type=str, default=None,
                    help=f"Combined dataset CSV (default: <dataset_dir>/{COMBINED_FILE}).")
parser.add_argument("-o", "--out_dir", type=str, default="results/figures",
                    help="Where the figures are written (default: results/figures).")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_labels(df: pd.DataFrame) -> tuple[int, int]:
    """Return (slicks, no-slicks). Rows without a label count as neither."""
    return int((df["slick_tag"] == True).sum()), int((df["slick_tag"] == False).sum())


def counts_frame(counts: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Turn {site: (true, false)} into a frame with totals and the positive rate."""
    df = pd.DataFrame(
        {
            "Site": list(counts),
            "True": [c[0] for c in counts.values()],
            "False": [c[1] for c in counts.values()],
        }
    )
    df["Total"] = df["True"] + df["False"]
    df["Percent"] = df["True"] / df["Total"]
    return df


def plot_counts(df: pd.DataFrame, title: str, out_path: Path) -> None:
    """One panel per site: the slick and no-slick bar, annotated with the share."""
    fig, axes = plt.subplots(1, len(df), figsize=(14, 4), sharey=True)
    axes = np.atleast_1d(axes)

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]
        bars = ax.bar(["True", "False"], [row["True"], row["False"]],
                      color=[COL_TRUE, COL_FALSE])
        for bar, share in zip(bars, [row["Percent"], 1 - row["Percent"]]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    f"{share:.2%}", ha="center", fontsize=12, weight="bold")
        ax.set_title(row["Site"], y=1.03)
        ax.set_xlabel("")
        ax.set_ylabel("Absolute frequencies" if i == 0 else "")
        ax.set_ymargin(0.1)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def month_pivot(df: pd.DataFrame, values: str) -> pd.DataFrame:
    """Sites as rows, calendar months as columns, sorted Jan-Dec."""
    pivot = df.pivot(index="site", columns="month", values=values).fillna(0)
    pivot.columns = [pd.Timestamp(year=2020, month=m, day=1).strftime("%b")
                     for m in pivot.columns]
    pivot = pivot.rename(index=SITE_LABELS).sort_index()
    return pivot[pivot.columns.sort_values(key=lambda c: pd.to_datetime(c, format="%b"))]


def plot_month_heatmap(pivot: pd.DataFrame, title: str, cbar_label: str,
                       fmt: str, out_path: Path) -> None:
    """Heatmap over site x month. Months a site never observed are left white."""
    plt.figure(figsize=(10, 3))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=sns.color_palette("viridis", as_cmap=True),
        mask=pivot == 0,
        cbar_kws={"label": cbar_label},
        linewidths=0.5,
        linecolor="gray",
        square=True,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("gray")
        spine.set_linewidth(0.7)

    plt.title(title)
    plt.xlabel("")
    plt.ylabel("")
    plt.yticks(rotation=0)
    plt.gca().set_facecolor("white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {dataset_dir.resolve()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- the per-site tables, before hourly aggregation ------------------------
    native = {}
    for label, file_name in NATIVE_FILES.items():
        path = dataset_dir / file_name
        if not path.is_file():
            print(f"  {file_name} not found, skipping the native-resolution figure")
            native = {}
            break
        native[label] = count_labels(pd.read_csv(path))

    if native:
        df_native = counts_frame(native)
        print("\nAt the sensors' own sampling rates:")
        print(df_native.to_string(index=False))
        plot_counts(df_native,
                    "Labels per site, at each sensor's own sampling rate",
                    out_dir / "slick_counts_native_resolution.png")

    # --- the combined hourly dataset ------------------------------------------
    combined_path = Path(args.combined) if args.combined else dataset_dir / COMBINED_FILE
    if not combined_path.is_file():
        raise SystemExit(f"Combined dataset not found: {combined_path.resolve()}")

    combined_df = pd.read_csv(combined_path)
    combined_df["Time"] = pd.to_datetime(combined_df["Time"])
    combined_df["month"] = combined_df["Time"].dt.month

    per_site = {
        label: count_labels(combined_df[combined_df["site"] == site])
        for site, label in SITE_LABELS.items()
    }
    df_combined = counts_frame(per_site)
    print("\nIn the combined hourly dataset:")
    print(df_combined.to_string(index=False))
    plot_counts(df_combined,
                "Labels per site (combined hourly dataset)",
                out_dir / "slick_counts_per_site.png")

    # --- seasonal distribution -------------------------------------------------
    tagged = combined_df[combined_df["slick_tag"].notnull()]
    slicks = tagged[tagged["slick_tag"] == True]

    counts = slicks.groupby(["site", "month"]).size().reset_index(name="count")
    pivot_counts = month_pivot(counts, "count")
    print("\nSlicks per site and month:")
    print(pivot_counts)
    plot_month_heatmap(pivot_counts, "Number of slicks per site and month",
                       "Number of slicks", ".0f",
                       out_dir / "slicks_per_site_per_month.png")

    totals = tagged.groupby(["site", "month"]).size().reset_index(name="total_samples")
    stats = totals.merge(counts.rename(columns={"count": "slick_true"}),
                         on=["site", "month"], how="left").fillna(0)
    stats["proportion"] = stats["slick_true"] / stats["total_samples"] * 100
    pivot_prop = month_pivot(stats, "proportion")
    print("\nShare of labelled samples that are slicks (%):")
    print(pivot_prop.round(1))
    plot_month_heatmap(pivot_prop, "Proportion of slicks per site and month",
                       "Proportion of slicks (%)", ".1f",
                       out_dir / "proportion_slicks_per_site_per_month.png")
