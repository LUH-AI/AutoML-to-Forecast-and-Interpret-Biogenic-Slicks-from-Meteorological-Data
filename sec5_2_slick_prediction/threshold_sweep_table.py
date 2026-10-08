r"""Sweep the hourly aggregation threshold of preprocess.py and print slick rates.

Takes labelled sub-hourly tables (sensors_with_labels.csv from the `merge` step) and, for each
threshold, reports the hourly slick rate and its distance from the sub-hourly rate.

    python sec5_2_slick_prediction/threshold_sweep_table.py \
        build/*/sensors_with_labels.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sec3_sensor_datasets.preprocess import aggregate_hour

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("inputs", nargs="+", help="Labelled sub-hourly CSVs, with Time and slick_tag columns.")
parser.add_argument("-t", "--thresholds", default="0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60",
                    help="Comma-separated thresholds to sweep (default: %(default)s).")
args = parser.parse_args()

THRESHOLDS = [float(x) for x in args.thresholds.split(",")]

for path in args.inputs:
    tags = pd.read_csv(path, parse_dates=["Time"]).set_index("Time")["slick_tag"]
    base = 100 * tags.eq(True).sum() / tags.notna().sum()
    binned = tags.resample("1h", label="right", closed="right")

    print("%s, sub-hourly rate %.2f %%" % (Path(path).parent.name, base))
    for threshold in THRESHOLDS:
        hourly = binned.apply(aggregate_hour, threshold=threshold)
        rate = 100 * hourly.eq(True).sum() / hourly.notna().sum()
        print("    threshold %.2f: rate %5.2f%%, delta %4.2f" % (threshold, rate, (rate - base)))
