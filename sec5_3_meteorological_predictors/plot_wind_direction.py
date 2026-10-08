r"""Wind-direction figures of the Shapley analysis (Section 5.3.2, Figures 12 and 15).

All three figures are polar bar charts with 16 direction sectors: bar length is the number of
samples whose wind direction falls into the sector, colour is what differs.

    slick-rate    P(slick | wind direction) from the windowed dataset, one panel per
                  timestep (11 and 12). No model is involved.
    data-vs-shap  Figure 12: for one timestep, the empirical slick rate (a, Data) next to the
                  model's mean total contribution phi_total (b, Model).
    shap-binned   Figure 15: the model's mean phi_total per sector, timesteps 11 and 12.

phi_total = phi(sin_t) + phi(cos_t) + phi(sin_t, cos_t) is the joint k-SII contribution of the
wind-direction encoding. The Shapley panels read explained_samples.csv and the
shapley_interactions_sample_*.json files of one run directory (calc_shap_interactions.py).

Run from the project root, as modules (they import from sec3_sensor_datasets):
    python -m sec5_3_meteorological_predictors.plot_wind_direction slick-rate \
        -d results/no_static/12h_combined_1hour_best/12h_combined_1hour.csv \
        -o results/figures/winddir_slick_rate_12h_combined.png
    python -m sec5_3_meteorological_predictors.plot_wind_direction data-vs-shap \
        -d results/no_static/12h_combined_1hour_best/12h_combined_1hour_test.csv \
        -s results/shapley_interactions/12h_combined_1hour_best \
        -o results/figures/wdir_data_vs_shap.png --timestep 12
    python -m sec5_3_meteorological_predictors.plot_wind_direction shap-binned \
        -d results/shapley_interactions/12h_combined_1hour_best -o results/figures
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable

from sec3_sensor_datasets.preprocess import decode_wind_direction

N_BINS = 16
DPI = 300
CMAP = plt.get_cmap("RdBu_r")
BAR_KW = dict(edgecolor="black", linewidth=0.4, alpha=0.9)
JSON_PATTERN = re.compile(r"shapley_interactions_sample_(\d+)\.json")

# Position of each wind-direction feature in the 6-feature frozen k-SII setup
# [winddir_sin_10, winddir_sin_11, winddir_sin_12, winddir_cos_10, winddir_cos_11, winddir_cos_12].
# CAREFUL: depends on the feature ordering used when the interaction JSONs were built.
WDIR_FEATURE_INDICES = {10: (0, 3), 11: (1, 4), 12: (2, 5)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def bin_by_direction(deg, n_bins=N_BINS):
    """Bin degrees into n_bins sectors centred on 0° = N, 22.5°, 45° = NE, ...
    Returns (centers_rad, bin_idx) with wrap-around, so no sample is dropped."""
    bw = 360 / n_bins
    idx = (np.floor(((deg + bw / 2) % 360) / bw).astype(int)) % n_bins
    return np.deg2rad(np.arange(0, 360, bw)), idx


def decode_timestep(df, t):
    """Degrees of timestep t's wind direction, without the rows where a component is NaN.
    Returns (degrees, mask) where mask selects the kept rows."""
    s = df[f"winddir_sin_{t}"].to_numpy()
    c = df[f"winddir_cos_{t}"].to_numpy()
    keep = ~(np.isnan(s) | np.isnan(c))
    return decode_wind_direction(s[keep], c[keep]) % 360, keep


def rate_by_direction(df, t, n_bins=N_BINS):
    """Empirical P(slick | direction) of timestep t.
    Returns centers, samples per bin, rate per bin and the base rate."""
    df = df[df["slick_tag"].notnull()]
    deg, keep = decode_timestep(df, t)
    y = df["slick_tag"].to_numpy().astype(bool)[keep]
    centers, idx = bin_by_direction(deg, n_bins)
    total = np.bincount(idx, minlength=n_bins)
    pos = np.bincount(idx[y], minlength=n_bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(total > 0, pos / total, np.nan)
    base = pos.sum() / total.sum() if total.sum() else 0.0
    return centers, total, rate, base


def load_interactions(shap_dir):
    """{sample_id: parsed json['data']} for the interaction files in shap_dir."""
    out = {}
    for path in Path(shap_dir).glob("shapley_interactions_sample_*.json"):
        m = JSON_PATTERN.match(path.name)
        if m:
            with open(path) as f:
                out[int(m.group(1))] = json.load(f)["data"]
    return out


def single_value(data, i):
    """Single-feature k-SII attribution phi(i). shapiq writes singletons as the stringified
    index ("0"); two other key formats are tried as a guard against a format change."""
    for key in (str(i), f"{i},", f"({i},)"):
        if key in data:
            return float(data[key])
    raise KeyError(f"No singleton entry for feature {i}; keys: {list(data.keys())[:8]}")


def total_contribution(data, i, j):
    """phi(i) + phi(j) + phi(i, j): the joint contribution of features i and j."""
    return single_value(data, i) + single_value(data, j) + float(data[f"{min(i, j)},{max(i, j)}"])


def add_phi_columns(samples, interactions, timesteps):
    """Add phi_t<t> (total wind-direction contribution) to the samples, per timestep, and drop
    the samples without an interaction file."""
    for t in timesteps:
        sin_idx, cos_idx = WDIR_FEATURE_INDICES[t]
        samples[f"phi_t{t}"] = samples["sample_id"].map(
            lambda sid: total_contribution(interactions[sid], sin_idx, cos_idx)
            if sid in interactions else np.nan)
    return samples.dropna(subset=[f"phi_t{t}" for t in timesteps])


def phi_by_direction(samples, t, n_bins=N_BINS):
    """Mean phi_total per direction bin of timestep t: centers, samples per bin, mean phi."""
    deg, keep = decode_timestep(samples, t)
    phi = samples[f"phi_t{t}"].to_numpy()[keep]
    centers, idx = bin_by_direction(deg, n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    mean_phi = np.array([phi[idx == i].mean() if np.any(idx == i) else 0.0 for i in range(n_bins)])
    return centers, counts, mean_phi


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def style_rose(ax, title, fs_title, pad, rlabel_pos, x_pad=None):
    """Compass-rose styling: N at the top, clockwise, ring labels only at 100 and 200."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_rlabel_position(rlabel_pos)
    ax.tick_params(axis="y", labelsize=8)
    if x_pad is not None:
        ax.tick_params(axis="x", pad=x_pad)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{int(v)}" if int(round(v)) in (100, 200) else ""))
    plt.setp(ax.get_yticklabels(), va="center")  # sit centred on the ring line
    ax.set_title(title, fontsize=fs_title, pad=pad)


def polar_axes(n, figsize):
    fig, axes = plt.subplots(1, n, subplot_kw={"polar": True}, figsize=figsize, dpi=DPI,
                             layout="constrained")
    return fig, np.atleast_1d(axes)


def share_radial_scale(axes, lower=None):
    """Give all panels the same radial limits so the bar lengths are comparable."""
    r_max = max(ax.get_ylim()[1] for ax in axes)
    r_min = min(ax.get_ylim()[0] for ax in axes) if lower is None else lower
    for ax in axes:
        ax.set_ylim(r_min, r_max)


def plot_slick_rate(df, title_prefix, out_path, timesteps=(11, 12)):
    """One polar panel per timestep of the empirical P(slick | wind direction), with a shared
    radial scale and a colourbar centred on the base rate."""
    fig, axes = polar_axes(len(timesteps), (7, 3))
    panels, rate_max = [], 0.0
    for t in timesteps:
        centers, total, rate, base = rate_by_direction(df, t)
        panels.append((t, centers, total, rate, base))
        if np.isfinite(rate).any():
            rate_max = max(rate_max, np.nanmax(rate))

    base_all = float(np.mean([p[4] for p in panels]))
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=base_all, vmax=max(rate_max, base_all + 1e-3))
    width = 2 * np.pi / N_BINS
    for ax, (t, centers, total, rate, _) in zip(axes, panels):
        colors = CMAP(norm(np.where(np.isfinite(rate), rate, base_all)))
        ax.bar(centers, total.astype(float), width=width, bottom=0.0, color=colors, **BAR_KW)
        style_rose(ax, f"Timestep {t}", fs_title=11, pad=12, rlabel_pos=45)

    r_max = max(ax.get_ylim()[1] for ax in axes) or 1.0
    for ax in axes:
        ax.set_ylim(0, r_max)

    sm = ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=list(axes), shrink=0.85, location="right", pad=0.05)
    cbar.set_label("P(slick | wind dir)", fontsize=10)
    cbar.ax.axhline(base_all, color="k", lw=1.4)
    fig.suptitle(f"{title_prefix} – wind-direction slick baseline (binned {360 / N_BINS:g}°)",
                 fontsize=13, y=1.02)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_data_vs_shap(df, samples, t, out_path):
    """Figure 12: (a) the empirical slick rate and (b) the mean phi_total, per direction sector
    of timestep t. The bars have the same length in both panels, only the colour differs."""
    centers, counts, rate, base = rate_by_direction(df, t)
    centers_b, counts_b, mean_phi = phi_by_direction(samples, t)
    if not np.array_equal(counts, counts_b):
        print("WARNING: panel (a) and (b) bar counts differ -- the two inputs are "
              "not the same test set, so the bars are not directly comparable.")

    # The data diverge about the base rate, phi about 0.
    norm_rate = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=base, vmax=max(np.nanmax(rate), base + 1e-3))
    vmax = max(float(np.max(np.abs(mean_phi))), 1e-6)
    norm_phi = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, (ax_l, ax_r) = polar_axes(2, (7.2, 3.5))
    width = 2 * np.pi / N_BINS
    ax_l.bar(centers, counts, width=width,
             color=CMAP(norm_rate(np.where(np.isfinite(rate), rate, base))), **BAR_KW)
    ax_r.bar(centers_b, counts_b, width=width, color=CMAP(norm_phi(mean_phi)), **BAR_KW)
    style_rose(ax_l, "(a) Data", fs_title=10, pad=9, rlabel_pos=60, x_pad=-2)
    style_rose(ax_r, "(b) Model", fs_title=10, pad=9, rlabel_pos=60, x_pad=-2)
    share_radial_scale([ax_l, ax_r], lower=0)

    for ax, norm, label, mark in [(ax_l, norm_rate, "P(slick | wind dir)", base),
                                  (ax_r, norm_phi, r"Mean $\varphi_{\mathrm{total}}$", None)]:
        sm = ScalarMappable(norm=norm, cmap=CMAP)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, location="bottom", shrink=0.85, pad=0.04, aspect=30)
        cbar.set_label(label, fontsize=9)
        if mark is not None:
            cbar.ax.axvline(mark, color="k", lw=1.2)  # the base rate
    fig.suptitle(rf"Wind Direction at Timestep {t}", fontsize=12)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_shap_binned(samples, timesteps, out_path):
    """Figure 15: the mean phi_total per direction sector, one polar panel per timestep, with a
    shared radial scale and a symmetric colourbar."""
    binned = {t: phi_by_direction(samples, t) for t in timesteps}
    vmax = max(max(float(np.max(np.abs(b[2]))) for b in binned.values()), 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axes = polar_axes(len(timesteps), (7, 3))
    width = 2 * np.pi / N_BINS
    for ax, t in zip(axes, timesteps):
        centers, counts, mean_phi = binned[t]
        ax.bar(centers, counts, width=width, color=CMAP(norm(mean_phi)), **BAR_KW)
        style_rose(ax, f"Timestep {t}", fs_title=11, pad=12, rlabel_pos=45)
    share_radial_scale(axes)

    sm = ScalarMappable(cmap=CMAP, norm=norm)
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.85, location="right", pad=0.05)
    cbar.set_label("Mean $φ_{total}$ per Bin", fontsize=10)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def read_samples(shap_dir, timesteps):
    """explained_samples.csv of a Shapley run with the phi_t<t> columns attached."""
    shap_dir = Path(shap_dir)
    samples_csv = shap_dir / "explained_samples.csv"
    if not samples_csv.exists():
        raise FileNotFoundError(f"{samples_csv} not found; run calc_shap_interactions.py first.")
    samples = pd.read_csv(samples_csv)
    interactions = load_interactions(shap_dir)
    print(f"Loaded {len(samples)} samples and {len(interactions)} interaction JSONs")
    samples = add_phi_columns(samples, interactions, timesteps)
    print(f"After matching JSONs: {len(samples)} samples")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="figure", required=True)

    p = sub.add_parser("slick-rate", help="empirical slick rate by wind direction")
    p.add_argument("-d", "--data_path", required=True, help="Windowed dataset CSV with slick_tag + winddir columns")
    p.add_argument("-t", "--title", default="12 Hours Windowed Dataframe")
    p.add_argument("-o", "--out_path", required=True, help="Where to save the figure")

    p = sub.add_parser("data-vs-shap", help="Figure 12: data next to the model's contribution")
    p.add_argument("-d", "--data_path", required=True, help="Windowed dataset CSV with slick_tag + winddir columns (panel a)")
    p.add_argument("-s", "--shap_dir", required=True, help="Folder with explained_samples.csv and the JSONs (panel b)")
    p.add_argument("-o", "--out_path", required=True, help="Where to save the figure")
    p.add_argument("--timestep", type=int, default=12, help="Window timestep to plot (default 12 = most recent)")

    p = sub.add_parser("shap-binned", help="Figure 15: the model's contribution, timesteps 11 and 12")
    p.add_argument("-d", "--data_path", required=True, help="Folder with explained_samples.csv and the JSONs")
    p.add_argument("-o", "--output_dir", required=True, help="Folder to write the figure into")

    args = parser.parse_args()
    if args.figure == "slick-rate":
        plot_slick_rate(pd.read_csv(args.data_path), args.title, args.out_path)
    elif args.figure == "data-vs-shap":
        if args.timestep not in WDIR_FEATURE_INDICES:
            raise ValueError(f"--timestep must be one of {sorted(WDIR_FEATURE_INDICES)} "
                             "(matching the frozen wind-direction features).")
        samples = add_phi_columns(pd.read_csv(Path(args.shap_dir) / "explained_samples.csv"),
                                  load_interactions(args.shap_dir), [args.timestep])
        plot_data_vs_shap(pd.read_csv(args.data_path), samples, args.timestep, args.out_path)
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timesteps = [11, 12]
        plot_shap_binned(read_samples(args.data_path, timesteps), timesteps, out_dir / "shap_wdir_total_binned.png")


if __name__ == "__main__":
    main()
