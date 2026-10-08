"""Definitions shared by every script in sec5_2_slick_prediction/.

Folder-name parsing, the wind-speed baseline and the tolerance-adjusted F1 live here so
a change is made once and every figure and table built from them stays consistent.

    Naming        parse_folder, label_to_minutes, label_to_display
    Result files  seed_prediction_csvs, load_full_gt_true_pos
    Baseline      find_wspd_col, wind_baseline_pred, wind_baseline_score
    Metric        adjusted_f1, has_match

Imported only, never run. Scripts are executed as files from the project root, e.g.
`python sec5_2_slick_prediction/plot_pr_curves.py`, which puts this folder on the import path.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Result-folder naming
# ---------------------------------------------------------------------------
# Folder name : {input_label}_{site}_{sample_rate}_{preset}
# CSV name    : {input_label}_{site}_{sample_rate}.csv
# These mirror the conventions in sec5_1_experimental_setup/generate_jobs.py, which writes them.
SITES = ["fino1_2022", "spiekeroog_2024", "spiekeroog_2021", "combined"]
SAMPLE_RATES = ["1hour"]
PRESETS = ["best"]

# Wind-speed rule from the literature: at or below this speed, predict a slick.
WSPD_THRESHOLD = 5.0  # m/s

_WSPD_RE = re.compile(r"^windspeed_(\d+)$")


def parse_folder(name: str) -> tuple[str, str, str]:
    """Folder name -> (input_label, site, sample_rate).

    '12h_combined_1hour_best' -> ('12h', 'combined', '1hour')

    Raises:
        ValueError: if the name does not match the expected pattern.
    """
    rest = name
    for preset in PRESETS:
        if rest.endswith(f"_{preset}"):
            rest = rest[: -(len(preset) + 1)]
            break
    for sr in SAMPLE_RATES:
        if rest.endswith(f"_{sr}"):
            rest = rest[: -(len(sr) + 1)]
            sample_rate = sr
            break
    else:
        raise ValueError(f"Cannot find sample rate in folder {name!r}")
    for site in SITES:
        if rest.endswith(f"_{site}"):
            rest = rest[: -(len(site) + 1)]
            site_name = site
            break
    else:
        raise ValueError(f"Cannot find site in folder {name!r}")
    return rest, site_name, sample_rate


def label_to_minutes(label: str) -> int:
    """'10min' -> 10, '12h' -> 720, '4days' -> 5760, '1week' -> 10080.

    Used to order the input lengths, which do not sort correctly as strings.
    """
    if label.endswith("min"):
        return int(label.removesuffix("min"))
    if label.endswith("h"):
        return int(label.removesuffix("h")) * 60
    if label.endswith("days"):
        return int(label.removesuffix("days")) * 24 * 60
    if label == "1week":
        return 7 * 24 * 60
    if label.endswith("weeks"):
        return int(label.removesuffix("weeks")) * 7 * 24 * 60
    raise ValueError(f"Unrecognised input label: {label!r}")


def label_to_display(label: str) -> str:
    """'12h' -> '12 hours', for the axis ticks."""
    def _unit(value: int, singular: str) -> str:
        return f"{value} {singular}" + ("" if value == 1 else "s")

    if label.endswith("min"):
        return _unit(int(label.removesuffix("min")), "minute")
    if label.endswith("h"):
        return _unit(int(label.removesuffix("h")), "hour")
    if label.endswith("days"):
        return _unit(int(label.removesuffix("days")), "day")
    if label == "1week":
        return "1 week"
    if label.endswith("weeks"):
        return _unit(int(label.removesuffix("weeks")), "week")
    raise ValueError(f"Unrecognised input label: {label!r}")


# ---------------------------------------------------------------------------
# Reading a result folder
# ---------------------------------------------------------------------------
def seed_prediction_csvs(folder: Path, input_label: str, site: str,
                         sample_rate: str) -> list[Path]:
    """The per-seed prediction CSVs of one model folder, in seed order."""
    return sorted(folder.glob(f"{input_label}_{site}_{sample_rate}_seed*_predictions.csv"))


def load_full_gt_true_pos(folder: Path, input_label: str, site: str,
                          sample_rate: str) -> np.ndarray:
    """Sorted timestamps of every slick in the full (train + test) ground truth.

    This is the windowed dataset train_tabular_predictor.py writes before splitting,
    so it holds the neighbours of a test row that ended up in the training split.
    """
    gt_path = folder / f"{input_label}_{site}_{sample_rate}.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"Full ground-truth file not found: {gt_path}")
    gt = pd.read_csv(gt_path, usecols=["Time", "slick_tag"])
    times = pd.to_datetime(gt.loc[gt["slick_tag"] == 1, "Time"]).to_numpy()
    return np.sort(times)


# ---------------------------------------------------------------------------
# Wind-speed baseline
# ---------------------------------------------------------------------------
def find_wspd_col(df: pd.DataFrame, *, required: bool = True) -> str | None:
    """The wind-speed column of the reading immediately before the prediction.

    Prediction CSVs carry one windspeed_<n> column, the last reading of the input
    window. If several are present, the highest index (most recent) wins.

    Args:
        required: Raise if no windspeed_<n> column is present, instead of returning None.
    """
    matches = [(int(m.group(1)), c) for c in df.columns if (m := _WSPD_RE.match(c))]
    if not matches:
        if required:
            raise KeyError(f"No windspeed_<n> column found; columns are {list(df.columns)}")
        return None
    return max(matches)[1]


def wind_baseline_pred(df: pd.DataFrame, threshold: float = WSPD_THRESHOLD) -> np.ndarray:
    """Baseline hard predictions: slick whenever the most recent reading is <= threshold."""
    wspd = df[find_wspd_col(df)].to_numpy(dtype=float)
    return (wspd <= threshold).astype(int)


def wind_baseline_score(df: pd.DataFrame) -> np.ndarray:
    """Ranking score for the baseline: lower wind speed means more slick-like.

    Valid for ROC-AUC/PR-AUC; not a probability, so must not be passed to the Brier score.
    """
    return -df[find_wspd_col(df)].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Tolerance-adjusted F1
# ---------------------------------------------------------------------------
def has_match(query: np.ndarray, ref: np.ndarray, tolerance: pd.Timedelta) -> np.ndarray:
    """For each timestamp in query, is there a ref timestamp within tolerance?"""
    if len(query) == 0 or len(ref) == 0:
        return np.zeros(len(query), dtype=bool)
    idx = np.searchsorted(ref, query)
    left = np.clip(idx - 1, 0, len(ref) - 1)
    right = np.clip(idx, 0, len(ref) - 1)
    d_left = np.abs(query - ref[left])
    d_right = np.abs(query - ref[right])
    return (d_left <= tolerance) | (d_right <= tolerance)


def adjusted_f1(times: pd.Series, y_true: np.ndarray, y_pred: np.ndarray,
                full_true_pos: np.ndarray,
                tolerance: pd.Timedelta = pd.Timedelta("1h")) -> dict:
    """F1 that accepts a prediction landing within tolerance of a real slick.

    - A prediction y_pred(t)=1 counts as a true positive if any slick in full_true_pos
      lies in [t - tolerance, t + tolerance], and as a false positive otherwise.
    - A test label y_true(t)=1 counts as found if any positive prediction lies in
      [t - tolerance, t + tolerance], and as a false negative otherwise.

    Precision is measured against full_true_pos, recall against this test split's own
    predictions - different true-positive counts, so this is not the F1 of a single
    confusion matrix.

    Args:
        full_true_pos: Sorted reference timestamps, see load_full_gt_true_pos.

    Returns:
        dict: adj_f1, adj_precision, adj_recall and the four underlying counts.
    """
    times = pd.to_datetime(times).to_numpy()
    pred_pos = np.sort(times[y_pred == 1])
    test_true_pos = np.sort(times[y_true == 1])

    tp_pred = int(has_match(pred_pos, full_true_pos, tolerance).sum())
    fp = len(pred_pos) - tp_pred
    tp_true = int(has_match(test_true_pos, pred_pos, tolerance).sum())
    fn = len(test_true_pos) - tp_true

    precision = tp_pred / (tp_pred + fp) if (tp_pred + fp) else 0.0
    recall = tp_true / (tp_true + fn) if (tp_true + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"adj_f1": f1, "adj_precision": precision, "adj_recall": recall,
            "tp_pred": tp_pred, "fp": fp, "tp_true": tp_true, "fn": fn}
