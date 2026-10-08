r"""Evaluate a trained AutoGluon MultiModalPredictor model on a test set of images.

Usage (from the project root):
python -m sec5_2_slick_prediction.evaluate_image_predictions \
    --model_dir /path/to/saved/model \
    --data_dir /path/to/image/data \
    --predictions_path /path/to/save/predictions.csv

Notes:
- data_dir must contain site subfolders with the expected CSVs and images.
- Output CSV columns: prediction, image_path, site, ground_truth.
"""

from autogluon.multimodal import MultiModalPredictor
import argparse
from sec4_1_label_creation.image_data import load_split
import os

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    "-m",
    "--model_dir",
    required=True,
    type=str,
    help="Path to the trained AutoGluon model.",
)
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    required=True,
    help="Path to the image data directory, where all sites are present.",
)
parser.add_argument(
    "-p",
    "--predictions_path",
    type=str,
    required=True,
    help="Path to save the predictions CSV file.",
)
args = parser.parse_args()

# Check if the provided data_dir is valid
assert os.path.isdir(args.model_dir), f"{args.model_dir} (argument '--model_dir'/'-m') is not a directory!"
assert os.path.isdir(args.data_dir), f"{args.data_dir} (argument '--data_dir'/'-d') is not a directory!"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load the trained model
    predictor = MultiModalPredictor.load(args.model_dir)    # Load the test DataFrame

    test_df = load_split(args.data_dir, train=False)

    # Make predictions
    predictions = predictor.predict(test_df, as_pandas=True)
    predictions_df = predictions.to_frame(name="prediction")
    predictions_df.columns = ["prediction"]

    # Add the image_path column to the predictions_df DataFrame
    predictions_df["image_path"] = test_df["image_path"]
    # Add the site column to the predictions_df DataFrame
    predictions_df["site"] = test_df["site"]
    # Add the true label column to the predictions_df DataFrame
    predictions_df["ground_truth"] = test_df["label"]
    # Save predictions_df to CSV for later analysis
    predictions_df.to_csv(args.predictions_path, index=False)

    scores = predictor.evaluate(test_df, metrics=["accuracy", "f1", "roc_auc", "log_loss"])

    # Print the evaluation scores
    print("Evaluation scores:")
    for metric, score in scores.items():
        print(f"{metric}: {score}")