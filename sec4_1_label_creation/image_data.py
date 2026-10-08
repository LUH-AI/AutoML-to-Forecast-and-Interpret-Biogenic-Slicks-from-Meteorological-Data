"""Train and test splits of the image dataset, shared by train_image_model.py and
sec5_2_slick_prediction/evaluate_image_predictions.py.

Layout under the dataset root, one folder per site (a site can be nested, e.g. Fino2/cam02):

    <root>/<site>/train.csv, test.csv   one row of image paths, relative to <root>/<site>/
    <root>/<site>/list_of_labels.csv    image name -> True/False
    <root>/<site>/...                   the images
"""
import os

import pandas as pd


def find_csv_files(root, keyword):
    """CSV files whose name contains `keyword` ('train' or 'test'), searching top-down and not
    descending below a folder that has one."""
    found = []
    for folder, subfolders, files in os.walk(root):
        hits = sorted(f for f in files if f.endswith(".csv") and keyword in f)
        if hits:
            found.extend(os.path.join(folder, f) for f in hits)
            subfolders.clear()
    return found


def load_split(root, train=True):
    """One DataFrame with the train or test split of every site under `root`.

    Columns: image_path (full path of the image), site, label (bool).
    """
    frames = []
    for csv_file in find_csv_files(root, "train" if train else "test"):
        if os.path.getsize(csv_file) == 0:  # a site without a split (empty file) is skipped
            continue
        folder = os.path.dirname(csv_file)
        site = os.path.relpath(folder, root)
        # The first column holds the image names whether or not the header has an index cell.
        labels = pd.read_csv(os.path.join(folder, "list_of_labels.csv"), index_col=0).iloc[:, 0]
        labels = labels.astype(str).str.lower() == "true"

        paths = pd.read_csv(csv_file, header=None).T[0].astype(str)
        df = pd.DataFrame({"image_path": paths.str.replace(f"{site}/", "", regex=False).str.replace(" ", "")})
        names = df["image_path"].apply(lambda p: os.path.basename(p).split(".")[0])
        unlabelled = names[~names.isin(labels.index)]
        assert unlabelled.empty, f"Image {unlabelled.iloc[0]} not found in the labels of {folder}"
        df["label"] = names.map(labels).astype(bool)
        df["image_path"] = df["image_path"].apply(lambda p: os.path.join(root, site, p).replace("//", "/"))
        df["site"] = site
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
