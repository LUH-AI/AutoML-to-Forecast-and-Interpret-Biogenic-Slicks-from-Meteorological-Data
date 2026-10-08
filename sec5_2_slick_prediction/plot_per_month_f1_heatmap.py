r"""Per-month F1 heatmap for one model folder.

Breaks one model's test-set performance down by site and calendar month.

One seed, not an ensemble
    Reads one seed's predictions (--seed, default 0), so cells describe one concrete
    model, scored the same way as the results tables.

Site attribution
    By calendar year, through SITE_BY_YEAR. Works only because the three datasets don't
    overlap in time (Spiekeroog 2021, Fino1 2022, Spiekeroog 2024). A site sharing a
    year with an existing one would be silently merged into it, and the prediction
    CSVs carry no site column to fall back on.

Cell states, decided per site and month
    ok       >= MIN_SAMPLES rows and at least one positive label -> coloured by F1
    no_pos   >= MIN_SAMPLES rows but no positive label           -> masked
    too_few  fewer than MIN_SAMPLES rows                         -> masked

    Both masked states are drawn the same way, as an empty white cell, so the figure
    does not distinguish "no slicks to find" from "too little data".

Input   a results/no_static/<run>/ folder holding <stem>_seed<N>_predictions.csv
Output  one PNG (-o, default per_month_f1.png)

Usage:
    python sec5_2_slick_prediction/plot_per_month_f1_heatmap.py \
        results/no_static/12h_combined_1hour_best \
        -o results/figures/per_month_f1.png
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import seaborn as sns

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("folder", type=Path)
parser.add_argument("-o", "--output", type=Path, default=Path("per_month_f1.png"))
parser.add_argument("-s", "--seed", type=int, default=0,
                    help="Which seed's predictions to plot (default: 0)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
SITE_BY_YEAR = {
    2021: "Spiekeroog (2021)",
    2022: "Fino1 (2022)",
    2024: "Spiekeroog (2024)",
}
SITE_ORDER = ["Fino1 (2022)", "Spiekeroog (2021)", "Spiekeroog (2024)"]
MIN_SAMPLES = 20


def load_seed_predictions(folder: Path, seed: int = 0) -> pd.DataFrame:
    """Load one seed's test-set predictions and bin each row by site and month.

    Args:
        seed: Which seed to read (default 0).

    Returns:
        pd.DataFrame: one row per test sample, with Time, y_true, y_pred and the
            year/month/site each row is binned into. Time is the last reading of the
            input window rather than the labelled hour, so a row whose label falls in
            the first hour of a month is binned into the preceding one.
    """
    matches = sorted(folder.glob(f"*_seed{seed}_predictions.csv"))
    if not matches:
        raise FileNotFoundError(f"No seed-{seed} predictions in {folder}")

    df = pd.read_csv(matches[0])[["Time", "y_true", "y_pred"]]
    df["Time"] = pd.to_datetime(df["Time"])
    df["year"] = df["Time"].dt.year
    df["month"] = df["Time"].dt.month
    df["site"] = df["year"].map(SITE_BY_YEAR)
    return df


def per_month_f1(df: pd.DataFrame) -> pd.DataFrame:
    """Score each (site, month) cell, applying the ok/no_pos/too_few rule from the
    module docstring, and return one row per cell with its F1 (NaN if masked) and status."""
    rows = []
    for site in SITE_ORDER:
        sub_site = df[df["site"] == site]
        for month in range(1, 13):
            sub = sub_site[sub_site["month"] == month]
            n, n_pos = len(sub), int(sub["y_true"].sum())
            if n < MIN_SAMPLES:
                status, f1 = "too_few", np.nan
            elif n_pos == 0:
                status, f1 = "no_pos", np.nan
            else:
                status = "ok"
                f1 = f1_score(sub["y_true"], sub["y_pred"], zero_division=0)
            rows.append({"site": site, "month": month, "f1": f1, "n": n,
                         "n_pos": n_pos, "status": status})
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_heatmap(folder: Path, out_path: Path, seed: int = 0) -> None:
    """Load one seed's predictions, score every (site, month) cell, and save the heatmap."""
    df = load_seed_predictions(folder, seed)
    cells = per_month_f1(df)
    pivot_f1 = cells.pivot(index="site", columns="month", values="f1").reindex(SITE_ORDER)
    pivot_status = cells.pivot(index="site", columns="month", values="status").reindex(SITE_ORDER)
    pivot_f1.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]

    # colour only the real F1 cells; no_pos and too_few -> white (masked)
    mask = pivot_status.values != "ok"

    cmap = sns.color_palette("viridis", as_cmap=True)

    plt.figure(figsize=(10, 3))
    ax = sns.heatmap(
        pivot_f1,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=0, vmax=1,
        mask=mask,
        cbar_kws={'label': 'F1 score'},
        linewidths=0.5,
        linecolor='gray',
        square=True,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("gray")
        spine.set_linewidth(.7)

    model = " ".join(f.capitalize() for f in folder.name.removesuffix("_1hour_best").split("_"))
    plt.title(f"Per-month F1 of the {model} model")
    plt.xlabel("")
    plt.yticks(rotation=0)
    plt.ylabel("")
    plt.gca().set_facecolor("white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    plot_heatmap(args.folder, args.output, args.seed)