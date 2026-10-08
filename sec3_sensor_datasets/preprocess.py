r"""Preprocessing of the sensor data and the slick labels (Stages 2 and 3).
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Per site: the grid its images were captured on, the observation period and the height of
# the wind sensor in metres. The grid is also the sensor table's resolution.
SITES = {
    "fino1_2022": dict(grid="10min", start="2022-03-07", end="2022-10-29", wind_height=41),
    "spiekeroog_2021": dict(grid="1h", start="2021-01-01", end="2021-12-31", wind_height=12),
    "spiekeroog_2024": dict(grid="5min", start="2024-01-30", end="2024-06-10", wind_height=12),
}
FEATURES = ["windspeed", "airtemperature", "airpressure", "humidity", "winddir_sin", "winddir_cos"]

# Raw column names of the BSH FINO1 export (leading space included) and the Spiekeroog station.
FINO1_COLUMNS = {
    " WSPD_CUP_140_MC.Cup Anemometer.51.0 [m/s]": "windspeed",
    " WDIR_VANE_315deg.Wind Vane.91.0 [deg]": "winddirection",
    " DRYT.Thermometer.34.0 [degC]": "airtemperature",
    " ATMP.Barometer.21.0 [hPa]": "airpressure",
    " RELH.Hygrometer.34.0 [%]": "humidity",
}
SPIEKEROOG_COLUMNS = {
    "TSS_windspeed": "windspeed", "TSS_winddirection": "winddirection",
    "TSS_airtemperature": "airtemperature", "TSS_airpressure": "airpressure",
    "TSS_humidity": "humidity",
}
PANGAEA_COLUMNS = {"ff": "windspeed", "dd": "winddirection", "TTT": "airtemperature",
                   "PPPP": "airpressure", "RH": "humidity"}

TOLERANCE = "2min"      # largest accepted distance between a grid slot and a reading
FILL_HOURS = 3          # gaps up to six hours: three hours forward, three backward
SLICK_SHARE = 0.25      # an hour is slick if at least this share of its labelled readings is

ALIGNED_CSV = "sensors_with_filenames.csv"
TO_CLASSIFY_CSV = "to_classify.csv"
LABELLED_CSV = "sensors_with_labels.csv"


# ---------------------------------------------------------------------------
# Wind helpers
# ---------------------------------------------------------------------------
def get_adjusted_wspd(wspd_at_z, z, zr=0.1, roughness_length=1e-4):
    """Adjust wind speed from height z to reference height zr via a logarithmic wind profile.

    roughness_length default (1e-4 m) is for open sea, following Ossenbeck 2023.
    """
    return (wspd_at_z * np.log(zr / roughness_length) / np.log(z / roughness_length)).round(2)


def encode_wind_direction(degrees):
    """Encode a compass direction as a sine/cosine pair on the unit circle.

    Removes the 359°/0° discontinuity. Cosine is the north-south axis, sine the east-west
    one. NaN propagates to both components. Returns a list [sin, cos] so callers can unpack it:
        df["winddir_sin"], df["winddir_cos"] = encode_wind_direction(df["dir_deg"])
    """
    return [np.sin(np.radians(degrees)).round(2), np.cos(np.radians(degrees)).round(2)]


def decode_wind_direction(enc_sin, enc_cos):
    """Recover the compass direction in degrees [0, 360) from a sine/cosine pair (0° = North)."""
    angles_deg = np.degrees(np.arctan2(enc_sin, enc_cos)).round(2)
    return (angles_deg + 360) % 360


# ---------------------------------------------------------------------------
# Weather table (Section 3)
# ---------------------------------------------------------------------------
def select_nearest(df, grid, tolerance=TOLERANCE):
    """Put each sensor column on a regular grid by the nearest reading within a tolerance.

    Returns a frame with one row per grid point and the same columns as `df`.
    """
    tol = pd.Timedelta(tolerance)
    grid = pd.DatetimeIndex(grid)
    step = grid.to_series().diff().dropna().min()
    if step is not pd.NaT and tol > step / 2:
        raise ValueError(
            f"tolerance {tol} exceeds half the grid spacing ({step / 2}); neighbouring "
            "grid points would compete for the same reading"
        )

    df = df.sort_index()
    left = pd.DataFrame({"__grid": grid})
    out = pd.DataFrame(index=grid)
    out.index.name = df.index.name or "Time"
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            out[col] = np.nan
            continue
        right = pd.DataFrame({"__grid": series.index, col: series.to_numpy()})
        matched = pd.merge_asof(left, right, on="__grid", direction="nearest", tolerance=tol)
        out[col] = matched[col].to_numpy()
    return out


def read_raw_weather(path, site):
    """Read a raw weather export into the columns Time + windspeed, winddirection,
    airtemperature, airpressure, humidity.

    Fino1: the semicolon separated BSH export (~570 columns, '#' comment header).
    Spiekeroog: the station's semicolon export, or the tab-delimited PANGAEA download
    (recognised by its '/* ... */' metadata block and the short names ff, dd, TTT, PPPP, RH).
    """
    if site == "fino1_2022":
        header = pd.read_csv(path, sep=";", comment="#", nrows=0).columns
        missing = [c for c in FINO1_COLUMNS if c not in header]
        if missing:
            raise SystemExit(
                f"Column(s) not found in {path}:\n  " + "\n  ".join(map(repr, missing))
                + "\nAvailable columns:\n  " + "\n  ".join(map(str, header)))
        df = pd.read_csv(path, sep=";", comment="#", usecols=["Time", *FINO1_COLUMNS])
        return df.rename(columns=FINO1_COLUMNS)

    with open(path, encoding="utf-8") as f:
        pangaea = f.readline().startswith("/*")
    if not pangaea:
        df = pd.read_csv(path, sep=";").rename(columns={"Date/Time": "Time", **SPIEKEROOG_COLUMNS})
        return df[["Time", *SPIEKEROOG_COLUMNS.values()]]

    with open(path, encoding="utf-8") as f:
        first_row = next((i + 1 for i, line in enumerate(f) if line.strip() == "*/"), 0)
    df = pd.read_csv(path, sep="\t", skiprows=first_row)
    short = {c: re.sub(r"\s*\[.*\]$", "", c).strip() for c in df.columns}  # "ff [m/s]" -> "ff"
    rename = {c: PANGAEA_COLUMNS[s] for c, s in short.items() if s in PANGAEA_COLUMNS}
    rename.update({c: "Time" for c, s in short.items() if s == "Date/Time"})
    missing = [n for n in ["Time", *PANGAEA_COLUMNS.values()] if n not in rename.values()]
    if missing:
        raise SystemExit(f"Could not find {missing} in {path}.\nColumns: {list(df.columns)}")
    return df.rename(columns=rename)[["Time", *PANGAEA_COLUMNS.values()]]


def build_weather_table(raw, site):
    """Weather table of one site on its image grid.

    Puts every variable on the site's grid (nearest reading per slot), adjusts the wind speed
    to 0.1 m, encodes the wind direction as sine/cosine, fills gaps of up to six hours
    (three forward, three backward) and crops to the observation period.
    """
    cfg = SITES[site]
    df = raw.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    for col in df.columns.drop("Time"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["winddir_sin"], df["winddir_cos"] = encode_wind_direction(df["winddirection"].astype("float64"))
    df = df.drop(columns=["winddirection"]).set_index("Time").sort_index()

    end = pd.Timestamp(cfg["end"]) + pd.Timedelta(days=1)
    grid = pd.date_range(cfg["start"], end, freq=cfg["grid"], inclusive="left")
    df = select_nearest(df, grid)
    df["windspeed"] = df["windspeed"].apply(lambda w: get_adjusted_wspd(w, cfg["wind_height"]))

    limit = int(FILL_HOURS * pd.Timedelta("1h") / pd.Timedelta(cfg["grid"]))
    df = df.ffill(limit=limit).bfill(limit=limit)
    return df.loc[cfg["start"]:cfg["end"], FEATURES]


# ---------------------------------------------------------------------------
# Image labels onto the sensor readings (Stage 2)
# ---------------------------------------------------------------------------
def read_inventory(images_csv):
    """Image inventory with a 'file_name' column; the time is read out of YYYYMMDD_HHMMSS."""
    df = pd.read_csv(images_csv)
    assert "file_name" in df.columns, (
        f"{images_csv} has no 'file_name' column; columns are {list(df.columns)}")
    df = df[["file_name"]].copy()
    df["file_name"] = df["file_name"].astype(str).str.replace(r"\.[A-Za-z]+$", "", regex=True)
    df["slick_tag"] = None
    df["time"] = pd.to_datetime(df["file_name"].str.extract(r"(\d{8}_\d{6})")[0], format="%Y%m%d_%H%M%S")
    unparsed = int(df["time"].isna().sum())
    if unparsed:
        print(f"  {unparsed} file name(s) carried no YYYYMMDD_HHMMSS timestamp and were dropped")
    df = df.dropna(subset=["time"]).drop_duplicates("file_name")
    return df.sort_values("time").reset_index(drop=True)


def attach_filenames(sensor_df, inventory_df, tolerance):
    """Attach each image's file name to the sensor reading closest in time.

    Where several images fall closest to the same reading, the one nearest in time wins.
    """
    matched = pd.merge_asof(
        inventory_df.sort_values("time"),
        sensor_df[["Time"]].reset_index().rename(columns={"index": "sensor_idx"}),
        left_on="time", right_on="Time", direction="nearest", tolerance=tolerance,
    )
    dropped = matched["sensor_idx"].isna().sum()
    if dropped:
        print(f"  {dropped} image(s) had no sensor reading within {tolerance} and were left unmatched")
    matched = matched.dropna(subset=["sensor_idx"])

    matched["gap"] = (matched["time"] - matched["Time"]).abs()
    before = len(matched)
    matched = matched.sort_values("gap").drop_duplicates("sensor_idx", keep="first")
    if before - len(matched):
        print(f"  {before - len(matched)} image(s) shared a reading with a closer image and were dropped")

    sensor_df.loc[matched["sensor_idx"].astype(int), "file_name"] = matched["file_name"].values
    return sensor_df


def align_images(meteo_csv, images_csv, out_dir, wspd_column="windspeed", wspd_threshold=5.0,
                 day_start=6, day_end=20, tolerance_min=None):
    """Tag windy readings as no-slick, attach each image to its reading, clear the night, and
    list the images that still need a label (to_classify.csv)."""
    os.makedirs(out_dir, exist_ok=True)
    inventory = read_inventory(images_csv)
    print(f"1. inventory   : {len(inventory)} images listed in {os.path.basename(images_csv)}")

    # Wind rule: too windy for a slick to form, so no image is needed.
    sensor_df = pd.read_csv(meteo_csv)
    sensor_df["Time"] = pd.to_datetime(sensor_df["Time"])
    sensor_df = sensor_df.sort_values("Time").reset_index(drop=True)
    assert wspd_column in sensor_df.columns, (
        f"Wind-speed column {wspd_column!r} not found in {meteo_csv}.\n"
        "Set --wspd_column to one of:\n  " + "\n  ".join(map(str, sensor_df.columns)))
    sensor_df["slick_tag"] = np.where(sensor_df[wspd_column] > wspd_threshold, False, None)
    sensor_df["file_name"] = None
    n_windy = int((sensor_df["slick_tag"] == False).sum())
    print(f"2. wind rule   : {n_windy} readings above {wspd_threshold} m/s tagged no-slick")

    step = sensor_df["Time"].diff().mode()
    step = step.iloc[0] if len(step) else pd.Timedelta(minutes=10)
    tolerance = pd.Timedelta(minutes=tolerance_min) if tolerance_min is not None else step / 2
    sensor_df = attach_filenames(sensor_df, inventory, tolerance)
    print(f"3. join        : {int(sensor_df['file_name'].notna().sum())} readings carry an image "
          f"(grid step {step}, tolerance {tolerance})")

    # A reading the wind rule already tagged does not need the image model.
    redundant = sensor_df["slick_tag"].notna() & sensor_df["file_name"].notna()
    sensor_df.loc[redundant, "file_name"] = np.nan
    print(f"               : {int(redundant.sum())} image(s) dropped, the wind rule had already "
          "tagged their reading")

    # Outside daylight the images show nothing, so tag and file name are dropped.
    night = (sensor_df["Time"].dt.hour < day_start) | (sensor_df["Time"].dt.hour >= day_end)
    sensor_df.loc[night, ["slick_tag", "file_name"]] = np.nan
    sensor_df.to_csv(os.path.join(out_dir, ALIGNED_CSV), index=False)
    print(f"4. daylight    : {int(night.sum())} night readings cleared -> {ALIGNED_CSV}")

    # Images that still have no tag are what the classifier has to label.
    to_classify = sensor_df[sensor_df["file_name"].notna() & sensor_df["slick_tag"].isna()]
    to_classify[["file_name", "Time", "slick_tag"]].to_csv(os.path.join(out_dir, TO_CLASSIFY_CSV), index=False)
    print(f"5. to classify : {len(to_classify)} images -> {TO_CLASSIFY_CSV}")


def merge_predictions(sensor_df, pred_df):
    """Write each predicted label onto the sensor reading carrying that image.

    Returns the updated table and the number of labels that had no matching reading.
    """
    tags = pred_df.set_index("file_name")["slick_tag"]
    sensor_df["slick_tag"] = sensor_df["slick_tag"].astype(object)
    known = sensor_df["file_name"].isin(tags.index)
    sensor_df.loc[known, "slick_tag"] = sensor_df.loc[known, "file_name"].map(tags).values
    unmatched = len(set(tags.index) - set(sensor_df["file_name"].dropna()))
    return sensor_df, unmatched


def merge_labels(predictions_csv, out_dir, out_filename=LABELLED_CSV):
    """Merge the classifier's labels onto the aligned table and prune file names that are not
    backed by a label. Writes the site's finished label table."""
    sensor_df = pd.read_csv(os.path.join(out_dir, ALIGNED_CSV))
    sensor_df["Time"] = pd.to_datetime(sensor_df["Time"])

    # classify_images.py writes an image_name column; hand labels use the same layout.
    pred_df = pd.read_csv(predictions_csv).rename(columns={"image_name": "file_name"})
    missing = {"file_name", "slick_tag"} - set(pred_df.columns)
    assert not missing, f"{predictions_csv} is missing column(s): {', '.join(sorted(missing))}"

    before = int(sensor_df["slick_tag"].notna().sum())
    sensor_df, unmatched = merge_predictions(sensor_df, pred_df)
    after = int(sensor_df["slick_tag"].notna().sum())
    print(f"1. merge : {len(pred_df)} labels read, tagged readings {before} -> {after}")
    if unmatched:
        print(f"         {unmatched} label(s) had no matching reading and were skipped")

    # A file name should only survive where the tag actually came from that image.
    from_image = sensor_df["file_name"].isin(pred_df["file_name"])
    cleared = int((sensor_df["file_name"].notna() & ~from_image).sum())
    sensor_df.loc[~from_image, "file_name"] = np.nan
    print(f"2. prune : {cleared} file name(s) not backed by a label cleared")

    out_path = os.path.join(out_dir, out_filename)
    sensor_df.to_csv(out_path, index=False)
    n_true = int((sensor_df["slick_tag"] == True).sum())
    n_false = int((sensor_df["slick_tag"] == False).sum())
    print(f"\n{len(sensor_df)} readings -> {out_filename} "
          f"({n_true} slick, {n_false} no-slick, {len(sensor_df) - n_true - n_false} untagged)")


# ---------------------------------------------------------------------------
# Hourly tables (Section 4.2) and the combined dataset (Stage 3)
# ---------------------------------------------------------------------------
def aggregate_hour(series, threshold=SLICK_SHARE):
    """slick_tag of one hour: True if at least `threshold` of the labelled readings are True,
    else False if any reading is False, else NaN (nothing in the hour was labelled)."""
    labelled = series.notna().sum()
    if labelled and series.eq(True).sum() / labelled >= threshold:
        return True
    if series.eq(False).any():
        return False
    return np.nan


def resample_to_hourly(df, threshold=SLICK_SHARE):
    """Resample a time-indexed table with a slick_tag column to hours, each hour labelled by its
    end. Other columns keep their last value up to and including the full hour."""
    resampled = df.resample("1h", label="right", closed="right")
    hourly = resampled[df.columns.difference(["slick_tag"])].last()
    hourly["slick_tag"] = resampled["slick_tag"].apply(aggregate_hour, threshold=threshold)
    return hourly.reset_index()


def hourly_table(data_file, output_file):
    """sensors_with_labels.csv -> 1hour_<site>.csv."""
    df = pd.read_csv(data_file)
    df["Time"] = pd.to_datetime(df["Time"])
    # An hourly row aggregates several images, so a single file name would misdescribe it.
    df = df.set_index("Time").drop(columns=["file_name"], errors="ignore")
    resample_to_hourly(df).to_csv(output_file, index=False)


def combine_hourly(inputs, output_file):
    """Concatenate the per-site hourly tables (named <resolution>_<site>.csv, the site is
    parsed from the file name) into the combined dataset, sorted by time."""
    frames = []
    for path in map(Path, inputs):
        df = pd.read_csv(path)
        df["Time"] = pd.to_datetime(df["Time"])
        df["site"] = path.stem.split("_", 1)[1]  # 1hour_spiekeroog_2021 -> spiekeroog_2021
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True).sort_values(["Time", "site"])
    combined.to_csv(output_file, index=False)
    print(f"Wrote {len(combined)} rows from {len(frames)} sites to {output_file}")


def hourly_from_labels(labels_csv, build_dir):
    """Join the published hourly labels (Time, site, slick_tag) with each site's weather table,
    reduced to hours exactly as resample_to_hourly does. Writes build/1hour_<site>.csv."""
    labels = pd.read_csv(labels_csv, parse_dates=["Time"])
    missing = {"Time", "site", "slick_tag"} - set(labels.columns)
    assert not missing, f"{labels_csv} is missing column(s): {', '.join(sorted(missing))}"

    for site, site_labels in labels.groupby("site"):
        meteo_path = os.path.join(build_dir, site, "meteo.csv")
        assert os.path.isfile(meteo_path), f"{meteo_path} not found; run the weather step for {site} first"
        meteo = pd.read_csv(meteo_path, parse_dates=["Time"]).set_index("Time")
        hourly = meteo.resample("1h", label="right", closed="right").last()

        tags = site_labels.drop_duplicates("Time").set_index("Time")["slick_tag"]
        unmatched = int((~tags.index.isin(hourly.index)).sum())
        hourly["slick_tag"] = tags.reindex(hourly.index)

        out_path = os.path.join(build_dir, f"1hour_{site}.csv")
        hourly.reset_index().to_csv(out_path, index=False)
        n_true = int((hourly["slick_tag"] == True).sum())
        n_false = int((hourly["slick_tag"] == False).sum())
        print(f"{site}: {len(hourly)} hours -> {out_path} ({n_true} slick, {n_false} no-slick, "
              f"{len(hourly) - n_true - n_false} unlabelled); {unmatched} label(s) outside the weather period")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def make_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("weather", help="raw weather export -> meteo.csv")
    p.add_argument("-s", "--site", choices=list(SITES), required=True)
    p.add_argument("-f", "--raw_file", required=True,
                   help="Raw export: BSH FINO1 file, Spiekeroog station file, or PANGAEA tab file")
    p.add_argument("-w", "--wind_height", type=int, default=None,
                   help="Height of the wind sensor in metres (default: 41 for Fino1, 12 for Spiekeroog)")
    p.add_argument("-o", "--out_filename", required=True, help="Output CSV, e.g. build/<site>/meteo.csv")

    p = sub.add_parser("align", help="images + meteo.csv -> sensors_with_filenames.csv, to_classify.csv")
    p.add_argument("-m", "--meteo_csv", required=True, help="Sensor table of this site (weather step)")
    p.add_argument("-i", "--images", required=True,
                   help="Inventory CSV with a file_name column of YYYYMMDD_HHMMSS names")
    p.add_argument("-c", "--wspd_column", default="windspeed",
                   help="Wind-speed column the threshold is applied to (default: windspeed)")
    p.add_argument("--wspd_threshold", type=float, default=5.0,
                   help="Readings above this wind speed are tagged no-slick (default: 5.0)")
    p.add_argument("--day_start", type=int, default=6, help="First daylight hour, inclusive (default: 6)")
    p.add_argument("--day_end", type=int, default=20, help="First hour after daylight, exclusive (default: 20)")
    p.add_argument("--tolerance_min", type=float, default=None,
                   help="Largest accepted image-to-reading distance in minutes (default: half the time step)")
    p.add_argument("-o", "--out_dir", required=True, help="Site folder, e.g. build/spiekeroog_2021")

    p = sub.add_parser("merge", help="classifier labels -> sensors_with_labels.csv")
    p.add_argument("-p", "--predictions", required=True, help="Label CSV with file_name and slick_tag")
    p.add_argument("-o", "--out_dir", required=True, help=f"Site folder holding {ALIGNED_CSV}")
    p.add_argument("--out_filename", default=LABELLED_CSV, help=f"Output name (default: {LABELLED_CSV})")

    p = sub.add_parser("hourly", help="sensors_with_labels.csv -> 1hour_<site>.csv")
    p.add_argument("-d", "--data_file", required=True, help="Path to sensors_with_labels.csv")
    p.add_argument("-o", "--output_file", required=True, help="Output CSV, e.g. build/1hour_<site>.csv")

    p = sub.add_parser("combine", help="1hour_<site>.csv files -> combined dataset")
    p.add_argument("inputs", nargs="+", help="Per-site hourly CSVs, <resolution>_<site>.csv")
    p.add_argument("-o", "--output_file", required=True, help="Where the combined dataset is written")

    p = sub.add_parser("hourly-labels", help="published hourly labels + meteo.csv -> 1hour_<site>.csv")
    p.add_argument("-l", "--labels", required=True, help="Hourly label file (labels/hourly_labels.csv)")
    p.add_argument("-b", "--build_dir", default="build",
                   help="Folder with <site>/meteo.csv; the 1hour_<site>.csv files are written here")
    return parser


def main():
    args = make_parser().parse_args()
    if args.command == "weather":
        assert os.path.isfile(args.raw_file), f"{args.raw_file} (--raw_file/-f) is not a file!"
        if args.wind_height is not None:
            SITES[args.site]["wind_height"] = args.wind_height
        table = build_weather_table(read_raw_weather(args.raw_file, args.site), args.site)
        os.makedirs(os.path.dirname(args.out_filename) or ".", exist_ok=True)
        table.to_csv(args.out_filename, index=True)
        print(f"{len(table)} rows -> {args.out_filename}")
    elif args.command == "align":
        assert os.path.isfile(args.meteo_csv), f"{args.meteo_csv} (--meteo_csv) is not a file!"
        assert os.path.isfile(args.images), f"{args.images} (--images) is not a file!"
        align_images(args.meteo_csv, args.images, args.out_dir, args.wspd_column, args.wspd_threshold,
                     args.day_start, args.day_end, args.tolerance_min)
    elif args.command == "merge":
        assert os.path.isfile(os.path.join(args.out_dir, ALIGNED_CSV)), (
            f"{os.path.join(args.out_dir, ALIGNED_CSV)} not found. Run the align step first.")
        assert os.path.isfile(args.predictions), f"{args.predictions} (--predictions) is not a file!"
        merge_labels(args.predictions, args.out_dir, args.out_filename)
    elif args.command == "hourly":
        hourly_table(args.data_file, args.output_file)
    elif args.command == "combine":
        combine_hourly(args.inputs, args.output_file)
    elif args.command == "hourly-labels":
        assert os.path.isfile(args.labels), f"{args.labels} (--labels/-l) is not a file!"
        hourly_from_labels(args.labels, args.build_dir)


if __name__ == "__main__":
    main()
