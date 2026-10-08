"""
Train an image classifier with AutoGluon's MultiModalPredictor on a dataset organized by
per-site subfolders.

Expected dataset layout
    <dataset_root>/
        site_A/
            train.csv
            test.csv
            list_of_labels.csv
            images/...
        site_B/
            train.csv
            list_of_labels.csv
            images/...

CSVs are discovered and merged via image_data.py; image paths are
resolved relative to --data_dir and asserted to exist before training. fit() uses
presets="best_quality", standalone=True, clean_ckpts=True and a 7-day time_limit; all other
hyperparameters are AutoGluon defaults (none were manually tuned for the reported results).
Sets HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1 for offline Hugging Face behavior.

Usage:
    python sec4_1_label_creation/train_image_model.py -d /path/to/dataset_root
"""
from image_data import load_split
import argparse
import os
from autogluon.multimodal import MultiModalPredictor

# Tell Hugging Face to work offline
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-d", "--data_dir", type=str, required=True, help="Path to dataset folder where each site has its own subfolder and train.csv/test.csv/list_of_labels.csv as well as the corresponding images.")
args = parser.parse_args()

# Check if the provided data_dir is valid
assert os.path.isdir(args.data_dir), f"{args.data_dir} (argument '--data_dir'/'-d') is not a directory!"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train_df = load_split(args.data_dir)
    test_df = load_split(args.data_dir, train=False)

    print(f"Training on {len(train_df)} images")
    print(f"Testing on {len(test_df)} images")

    # AutoGluon only needs the image path and the label
    train_df = train_df.drop(columns=["site"])
    test_df = test_df.drop(columns=["site"])

    # Check whether all paths are correct
    for img in train_df["image_path"]:
        assert os.path.isfile(img), f"Image {img} not found in dataset!"

    # Predict using AutoGluon
    predictor = MultiModalPredictor(label="label", problem_type="binary", eval_metric="f1").fit(
        train_data=train_df,
        time_limit=60*60*168,  # 7 days
        standalone=True,
        clean_ckpts=True,
        presets="best_quality",
    )

    predictions = predictor.predict(test_df.drop(columns="label"))
    scores = predictor.evaluate(test_df, metrics=["accuracy", "f1", "roc_auc", "log_loss"])
    # Print the evaluation scores
    print("Evaluation scores:")
    for metric, score in scores.items():
        print(f"{metric}: {score}")