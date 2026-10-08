"""
Loads a trained AutoGluon MultiModalPredictor and predicts slicks for every .jpg image in a
directory, writing predictions.csv to the current working directory.

Usage:
    python sec4_1_label_creation/classify_images.py -m /path/to/saved_model -d /path/to/images
"""
from autogluon.multimodal import MultiModalPredictor
import pandas as pd
import argparse
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
    help="Path to the image data directory.",
)

args = parser.parse_args()

# Check if the provided data_dir is valid
assert os.path.isdir(args.model_dir), f"{args.model_dir} (argument '--model_dir'/'-m') is not a directory!"
assert os.path.isdir(args.data_dir), f"{args.data_dir} (argument '--data_dir'/'-d') is not a directory!"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load the trained model
    predictor = MultiModalPredictor.load(args.model_dir)

    # Create the DataFrame for all images in the data directory
    df = pd.DataFrame(columns=["image_name", "image_path"])

    # Iterate over all images in the data directory and add them to the DataFrame
    for f in os.listdir(args.data_dir):
        if f.endswith(".jpg"):
            df.loc[len(df)] = [f.replace(".jpg", ""), os.path.join(args.data_dir, f)]


    # Make predictions
    predictions = predictor.predict(df.drop(columns="image_name"), as_pandas=True)

    df["slick_tag"] = predictions
    df.drop(columns="image_path", inplace=True)
    df.to_csv("predictions.csv", index=False)
