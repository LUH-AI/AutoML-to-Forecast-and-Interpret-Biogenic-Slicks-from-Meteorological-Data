"""
Loads a trained AutoGluon TabularPredictor, predicts on a test CSV, and saves the result.

Output CSV columns
- prediction: the model's prediction for each row.
- ground_truth: the test CSV's label column, if it matches predictor.label.
- Time: re-attached from the input CSV, if present.

Usage:
    python sec5_1_experimental_setup/forecast_slick.py -m /path/to/autogluon_model -t /path/to/test.csv -p /path/to/predictions.csv
"""
from autogluon.tabular import TabularPredictor
import pandas as pd
import argparse
import os

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-m","--model_dir", required=True, type=str, help="Path to the trained AutoGluon model.")
parser.add_argument("-t","--test_csv", required=True, type=str, help="Path to the test CSV file.")
parser.add_argument("-p","--predictions_path", required=True, type=str, help="Path to save the predictions CSV file.")
args = parser.parse_args()

assert os.path.isdir(args.model_dir), f"{args.model_dir} (argument '--model_dir'/'-m') is not a directory!"
assert os.path.isfile(args.test_csv), f"{args.test_csv} (argument '--test_csv'/'-t') is not a file!"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load model and test data
    predictor = TabularPredictor.load(args.model_dir)
    test_df = pd.read_csv(args.test_csv)

    # Make predictions
    pred_df = predictor.predict(test_df.drop(columns=["Time"]), as_pandas=True).to_frame(name="prediction")

    # Add ground truth column if present
    if predictor.label in test_df.columns:
        pred_df["ground_truth"] = test_df[predictor.label]

    # Add Time column back if present
    pred_df["Time"] = test_df["Time"]

    # Save predictions
    if args.predictions_path:
        pred_df.to_csv(args.predictions_path, index=False)