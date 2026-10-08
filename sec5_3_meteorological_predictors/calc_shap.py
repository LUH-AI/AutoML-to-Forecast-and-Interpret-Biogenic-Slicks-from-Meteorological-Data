r"""
First-order Kernel SHAP values for a trained AutoGluon tabular slick model.

One Shapley value per feature per explained sample, on the predicted P(slick) scale —
input to the feature-importance and dependence plots. The sine/cosine wind-direction
pair needs pairwise interactions on top; that's handled separately by
calc_shap_interactions.py.

Method
    The model is wrapped so SHAP sees a plain f(X) -> P(class=1), and explained with
    shap.KernelExplainer against a background of --n_background rows drawn from the
    TRAINING split (default 200, fixed random_state). Per-sample contributions sum to
    P(slick | x) - mean P(slick | background).

    Two settings matter:
      nsamples=20000  coalitions sampled per explained sample, well above shap's default
                      of 2*n_features+2048 — this is why the run needs the array below.
      l1_reg=False    disables pre-selection of only "num_features(10)", which would
                      keep 10 of the 72 features and silently zero the rest.

Parallelism
    The array splits the explained SAMPLES of ONE model into --num_jobs chunks (not
    models). The sample set is fixed before splitting, so the union of all chunks
    equals a single-job run — but only if every chunk lands; nothing downstream checks
    for a missing one, so verify shap_values_job_*.npy count equals --num_jobs before
    plotting.

Outputs, written to --output_dir (suffix _job_<NNN> when --num_jobs > 1)
    shap_values<suffix>.npy   (n_explained, n_features) float array
    X_explain<suffix>.csv     sample_id, original_index, the feature values, y_true
    expected_value.txt        background mean prediction   (job 0 only)
    feature_names.txt         feature order of the .npy    (job 0 only)

    plot_shap.py concatenates the chunks in job order and renders the summary and
    dependence plots.

Usage:
    python sec5_3_meteorological_predictors/calc_shap.py \
        -m AutogluonModels/12h_combined_1hour_best \
        -b results/no_static/12h_combined_1hour_best/12h_combined_1hour_train.csv \
        -d results/no_static/12h_combined_1hour_best/12h_combined_1hour_test.csv \
        -o results/shapley_values/12h_combined_1hour_best \
        --job_index 0 --num_jobs 100
"""
import numpy as np
import pandas as pd
import shap
from autogluon.tabular import TabularPredictor
from pathlib import Path
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
import argparse


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-b", "--train_data_path", type=str, required=True, help="Path to csv file that contains the train data")
parser.add_argument("-d", "--test_data_path", type=str, required=True, help="Path to csv file that contains the test data")
parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path to directory, where the SHAP outputs should be saved.")
parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the trained AutoGluon model directory")
parser.add_argument("-s", "--sample_size", type=int, required=False, default=None, help="Number of test samples to explain. If omitted, all test samples are used.")
parser.add_argument("-n", "--n_background", type=int, default=200, help="Number of random background samples drawn from the training set (default 200).")
parser.add_argument("--job_index", type=int, default=0, help="0-based index of this job within the array (default 0).")
parser.add_argument("--num_jobs", type=int, default=1, help="Total number of jobs in the array. The selected samples are split into this many chunks; this job processes chunk job_index.")
args = parser.parse_args()

if args.num_jobs < 1:
    raise ValueError("--num_jobs must be >= 1")
if not (0 <= args.job_index < args.num_jobs):
    raise ValueError(f"--job_index must be in [0, {args.num_jobs}), got {args.job_index}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def print_debug(msg):
    """Print a progress/diagnostic line, bracketed for visibility in SLURM logs."""
    print(f"***** {msg} *****")


# --------------------------------------------------------------------------- #
# Model adapter
# --------------------------------------------------------------------------- #
class AutoGluonAdapter:
    """
    Adapter for AutoGluon TabularPredictor binary classification, compatible with shap.

    Returns P(class=1) as a 1D numpy array.
    """

    def __init__(self, predictor, feature_names):
        """
        Parameters:
        -----------
        predictor : TabularPredictor
            The trained AutoGluon predictor
        feature_names : list of str
            Feature names, ordered to match the training data
        """
        self.predictor = predictor
        self.feature_names = feature_names
        self._validate_features()

    def _validate_features(self):
        """Warn if feature_names doesn't match the predictor's expected features."""
        try:
            model_features = self.predictor.feature_metadata_in.get_features()
            if set(self.feature_names) != set(model_features):
                print_debug("WARNING: Feature mismatch detected!")
                print_debug(f"Provided not in model: {set(self.feature_names) - set(model_features)}")
                print_debug(f"Missing from provided: {set(model_features) - set(self.feature_names)}")
        except Exception as e:
            print_debug(f"Could not validate features: {e}")

    def __call__(self, X):
        """
        Predict function compatible with shap.KernelExplainer.

        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Input features of shape (n_samples, n_features)

        Returns:
        --------
        np.ndarray
            1D array of probabilities P(class=1) with shape (n_samples,)
        """
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        elif isinstance(X, pd.DataFrame):
            X_df = X[self.feature_names].copy()
        else:
            raise TypeError(f"Expected np.ndarray or pd.DataFrame, got {type(X)}")

        proba = self.predictor.predict_proba(X_df, as_multiclass=True)

        if isinstance(proba, pd.DataFrame):
            if 1 in proba.columns:
                y_pred = proba[1].values
            elif '1' in proba.columns:
                y_pred = proba['1'].values
            else:
                y_pred = proba.iloc[:, -1].values
        elif isinstance(proba, pd.Series):
            y_pred = proba.values
        else:
            y_pred = np.array(proba).flatten()

        return np.asarray(y_pred).flatten()

    def predict_proba(self, X):
        """Alias for __call__ to maintain compatibility."""
        return self(X)


# --------------------------------------------------------------------------- #
# Load model and data
# --------------------------------------------------------------------------- #
predictor = TabularPredictor.load(args.model_path)

train_df = pd.read_csv(args.train_data_path)
test_df = pd.read_csv(args.test_data_path)

X_train = train_df.drop(columns=['slick_tag', 'Time'])
X_test = test_df.drop(columns=['slick_tag', 'Time'])
y_test = test_df["slick_tag"]

assert list(X_train.columns) == list(X_test.columns)
feature_names = X_train.columns.tolist()
print(f"Loaded {len(train_df)} train / {len(test_df)} test samples, {len(feature_names)} features")

model_wrapper = AutoGluonAdapter(predictor=predictor, feature_names=feature_names)

n_bg = min(args.n_background, len(X_train))
background = X_train.sample(n=n_bg, random_state=42).values
print_debug(f"Background distribution: {background.shape}")

# --------------------------------------------------------------------------- #
# Select samples to explain
# --------------------------------------------------------------------------- #
# Deterministically pick the global set of samples to explain.
# Identical across all jobs (fixed seed) so chunks together reproduce a single-job run.
if args.sample_size is not None:
    all_to_explain_idx = np.random.default_rng(42).choice(len(X_test), size=args.sample_size, replace=False)
else:
    all_to_explain_idx = np.arange(len(X_test))

# Split into num_jobs chunks; take ours.
global_positions = np.array_split(np.arange(len(all_to_explain_idx)), args.num_jobs)[args.job_index]
to_explain_idx = all_to_explain_idx[global_positions]
X_explain = X_test.iloc[to_explain_idx].reset_index(drop=True)

print(f"Job {args.job_index}/{args.num_jobs}: explaining {len(X_explain)} of {len(all_to_explain_idx)} samples "
      f"(global sample_ids {global_positions[0]}..{global_positions[-1]})")

OUT_DIR = Path(args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Compute SHAP values
# --------------------------------------------------------------------------- #
explainer = shap.KernelExplainer(model_wrapper, background)
shap_values = explainer.shap_values(X_explain.values, nsamples=20000, l1_reg=False)

# --------------------------------------------------------------------------- #
# Save outputs
# --------------------------------------------------------------------------- #
# Per-job outputs. plot_shap.py concatenates the chunks afterwards.
if args.num_jobs == 1:
    suffix = ""
else:
    suffix = f"_job_{args.job_index:03d}"

np.save(OUT_DIR / f"shap_values{suffix}.npy", shap_values)
X_explain_out = X_explain.copy()
X_explain_out.insert(0, "sample_id", global_positions)
X_explain_out.insert(1, "original_index", to_explain_idx)
X_explain_out["y_true"] = y_test.iloc[to_explain_idx].values
X_explain_out.to_csv(OUT_DIR / f"X_explain{suffix}.csv", index=False)

# Expected value is identical across jobs (same background, same model) -> write once.
if args.job_index == 0:
    with open(OUT_DIR / "expected_value.txt", "w") as f:
        f.write(str(explainer.expected_value))
    with open(OUT_DIR / "feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))

print(f"Saved chunk under {OUT_DIR}/ (job {args.job_index}/{args.num_jobs})")