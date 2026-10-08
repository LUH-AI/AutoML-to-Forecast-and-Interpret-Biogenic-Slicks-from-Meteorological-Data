r"""Window a site's hourly sensor table and train an AutoGluon-Tabular slick forecaster.

Turns the time series into a plain tabular problem: slide a window of --window_size rows
over the data, flatten it into one feature vector, and pair it with the label of the row
immediately after the window. A longer window is more columns, not a different model.

    1. Load the hourly CSV.
    2. Build the windows and write the flattened dataset to --output.
    3. Split 70/30 and write both halves next to it.
    4. Fit a TabularPredictor and write the test-set predictions.
    5. With --cleanup, delete the fitted model again.

Input
    One hourly table from sec3_sensor_datasets/, single-site or combined. Required columns:
    Time, slick_tag, windspeed, airtemperature, airpressure, humidity, winddir_sin/winddir_cos.

Split and seeds
    train_test_split with test_size=0.3, stratified on slick_tag, random_state fixed at 42 —
    independent of --seed, so every seed of a given window length trains on the same rows.
    The spread across seeds then measures run-to-run variance of the fit alone.

Outputs, derived from --output (<base> is --output without its suffix)
    <base>.csv                         the flattened windowed dataset
    <base>_train.csv, <base>_test.csv  the 70/30 split, identical for every seed
    <base>_seed<N>_predictions.csv     Time, windspeed_<W>, y_true, y_pred, proba_0, proba_1

Usage:
    python sec5_1_experimental_setup/train_tabular_predictor.py \
        -d build/1hour_combined_dataset.csv \
        -w 12 -p best \
        -o results/no_static/12h_combined_1hour_best/12h_combined_1hour.csv \
        --seed 0 --cleanup
"""
from autogluon.tabular import TabularPredictor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-d", "--data", type=str, required=True, help="Path to csv file that contains the dataset")
parser.add_argument("-w", "--window_size", type=int, required=True, help="Size of the time window in number of entries.")
parser.add_argument("-o", "--output", type=str, help="Specify, if the training dataframe should be saved at the given path")
parser.add_argument("-p", "--preset", type=str, default="good",help="Specify the autogluon preset (see docs)")
parser.add_argument(
    "--debug", 
    action="store_true",
    dest="use_debug",
    help="Print progress messages")
parser.add_argument(
    "--model_dir",
    type=str,
    default=None,
    help=(
        "Where to save the fitted model. Left unset, AutoGluon picks "
        "AutogluonModels/ag-<timestamp>, which is fine for runs that pass --cleanup. "
        "Set it for a model you intend to keep, e.g. "
        "AutogluonModels/12h_combined_1hour_best, so the Shapley job scripts find it."
    ))
parser.add_argument(
    "--cleanup",
    action="store_true",
    dest="use_cleanup",
    help="Deletes the trained model"
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help=(
        "Random seed for model training (default: 0). "
        "The train/test split is always fixed at seed=42 regardless of this value, "
        "so different seeds give the same data split but different model initializations. "
    )
)
args = parser.parse_args()

# Checks if command-line arguments are valid
assert os.path.isfile(args.data), f"{args.data} (argument '--data'/'-d') is not a file!"
assert args.window_size > 0, "Window size must be greater than 0!"
assert args.preset in ['extreme', 'best', 'high', 'good', 'medium'], "Preset must be one of ['extreme', 'best', 'high', 'good', 'medium']"

DEBUG = args.use_debug

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_debug(msg):
    if DEBUG:
        print(f"***** {msg} *****")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load data
    print_debug("Loading data")
    df = pd.read_csv(args.data)
    df['Time'] = pd.to_datetime(df['Time'])

    # 2. Create windowed dataframe
    print_debug("Creating windowed dataframe")
    features = ['windspeed', 'airtemperature', 'airpressure', 'humidity',
                'winddir_sin', 'winddir_cos']



    j = 0
    rows = []
    labels = []
    times = []
    
    # Windows may span two sites in the combined file. Harmless here: each site starts and ends
    # with NaN rows (gaps over six hours are not imputed), and windows containing NaNs are dropped.
    for i in range(len(df) - args.window_size):
        window_time = df.iloc[i + args.window_size-1]['Time'] #time corresponds to the end of the input window
        
        # The label is the row immediately after the window (one-step-ahead forecast).
        label = df.iloc[i + args.window_size]['slick_tag']
        if pd.isna(label):
            continue

        window = df.iloc[i:i+args.window_size][features].values.T.flatten()
        if pd.isna(window).any():  # check for bad windows
            j += 1
            continue

        rows.append(window)
        labels.append(int(label))
        times.append(window_time)

    print_debug(f"Skipped {j} bad windows, because of NaN values.")

    col_names = []
    for f in features:
        for t in range(1, args.window_size+1):
            col_names.append(f"{f}_{t}")

    df_flat = pd.DataFrame(rows, columns=col_names)
    df_flat['slick_tag'] = labels
    df_flat['Time'] = times
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save the flattened dataframe
        df_flat.to_csv(output_path, index=False)
        
    # 3. Create train test split
    print_debug("Creating train and test split")
    label = 'slick_tag'
    train_data, test_data = train_test_split(df_flat, test_size=0.3, random_state=42, stratify=df_flat[label])

    print_debug(f"Shape of train data: {train_data.shape}")
    print_debug(f"Shape of test data: {test_data.shape}")
    if args.output:
        # Additionally save the train and test dataframes
        base = output_path.with_suffix('')
        train_data.to_csv(f"{base}_train.csv", index=False)
        test_data.to_csv(f"{base}_test.csv", index=False)

    # Preserve test times/last windspeed for the predictions CSV before dropping the column
    test_times = test_data['Time'].values
    test_wspds = test_data[f'windspeed_{args.window_size}'].values
    #Drop time column for training
    train_data.drop(columns=['Time'], inplace=True)
    test_data.drop(columns=['Time'], inplace=True)

    X_train = train_data.drop(columns=[label]).values
    y_train = train_data[label].values.astype(int)
    X_test = test_data.drop(columns=[label]).values
    y_test = test_data[label].values.astype(int)

    print_debug(f"  Train  : {X_train.shape}")
    print_debug(f"  Test   : {X_test.shape}")
    print_debug(f"  Classes: {np.unique(y_train)}  "
          f"(train balance: {np.bincount(y_train) / len(y_train)})")
    print()

    # 4. Train model (CPU only)
    print_debug("Training model")
    predictor = TabularPredictor(
        label=label, 
        eval_metric="f1",
        problem_type='binary',
        path=args.model_dir,
        learner_kwargs={'random_state': args.seed}
    ).fit(
        train_data,
        presets=args.preset,
        ds_args={"enable_ray_logging": False}
    )

    # 5. Evaluate model
    print_debug("Evaluating model")
    print_debug(f"Model random seed: {args.seed}")

    y_proba = predictor.predict_proba(test_data); 
    print(predictor.evaluate_predictions(test_data[label], y_proba))
    if args.output:
        preds_df = pd.DataFrame({
            "Time": test_times,
            f'windspeed_{args.window_size}':test_wspds,
            "y_true": y_test,
            "y_pred": predictor.predict(test_data).to_numpy(),
            "proba_0": y_proba.iloc[:, 0],
            "proba_1": y_proba.iloc[:, 1],
        })
        preds_path = f"{base}_seed{args.seed}_predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        print_debug(f"Saved predictions to {preds_path}")

    if args.use_cleanup:
        model_path = predictor.path
        try:
            shutil.rmtree(model_path)
            print_debug(f"Successfully deleted Autogluon model at: {model_path}")
        except:
            print(f"ERROR: Couldn't delete Autogluon model at: {model_path}")