r"""
Second-order Shapley interaction values for the wind-direction encoding.

Wind direction enters the model as a sine/cosine pair, so neither component alone
describes what the model does with the angle. This script computes k-SII interaction
values (max_order=2) so the two recombine into one directional contribution per
timestep:

    phi_total(t) = phi(sin_t) + phi(cos_t) + phi(sin_t, cos_t)

Players are the six wind-direction features of the three most recent timesteps; every
other feature is held at the explained sample's own observed value, so the attribution
reflects that sample's actual weather context. Six players means the 2^6 = 64
coalitions fit in the budget, so the values are exact rather than estimated.

!!! ONLY FOR THE 12h MODEL !!!
The player set is hardcoded to winddir_{sin,cos}_{10,11,12}, and the plotting scripts
map shapiq's player indices 0..5 back to those names *positionally*
(WDIR_FEATURE_INDICES in plot_wind_direction.py). This
fails silently on a model with a different window length:

  - shorter window (e.g. 3h): no _10/_11/_12 columns exist, so no feature is selected
    and the run writes one JSON per sample explaining nothing at all;
  - longer window (e.g. 24h, 4 weeks): the _10/_11/_12 columns exist but timestep 12
    sits mid-window, not at its end, so figures mislabel a mid-window hour as "the
    most recent reading".

Neither is checked at runtime — train and explain a 12h model, or adapt the player set
and both plotting scripts together.

Outputs, one set per array chunk, written to --output_dir:
    explained_samples[_job_<NNN>].csv        feature values, y_true and y_pred_proba
    shapley_interactions_sample_<id>.json    one shapiq InteractionValues per sample

The array parallelises the test SAMPLES of one model, not the models; sample_id is the
join key between the CSV and the JSON file names. Stitch the per-chunk CSVs together
afterwards (see sec5_3_meteorological_predictors/calculate_12h_shapiq.sh) before plotting.

Usage:
    python sec5_3_meteorological_predictors/calc_shap_interactions.py \
        -m AutogluonModels/12h_combined_1hour_best \
        -b results/no_static/12h_combined_1hour_best/12h_combined_1hour_train.csv \
        -d results/no_static/12h_combined_1hour_best/12h_combined_1hour_test.csv \
        -o results/shapley_interactions/12h_combined_1hour_best \
        --job_index 0 --num_jobs 50
"""
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from shapiq import Explainer
from pathlib import Path
import argparse


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-b", "--train_data_path", type=str, required=True, help="Path to csv file that contains the train data")
parser.add_argument("-d", "--test_data_path", type=str, required=True, help="Path to csv file that contains the test data")
parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path to directory, where the interaction values should be saved.")
parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the trained AutoGluon model directory")
parser.add_argument("-s", "--sample_size", type=int, required=False, default=None, help="Number of test samples to explain. If omitted, all test samples are used.")
parser.add_argument("--job_index", type=int, default=0, help="0-based index of this job within the array (default 0).")
parser.add_argument("--num_jobs", type=int, default=1, help="Total number of jobs in the array. The selected samples are split into this many chunks; this job processes chunk job_index. Default 1 reproduces the original single-job behaviour.")
args = parser.parse_args()

if args.num_jobs < 1:
    raise ValueError("--num_jobs must be >= 1")
if not (0 <= args.job_index < args.num_jobs):
    raise ValueError(f"--job_index must be in [0, {args.num_jobs}), got {args.job_index}")

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def print_debug(msg):
    print(f"***** {msg} *****")

# --------------------------------------------------------------------------- #
# Model adapter
# --------------------------------------------------------------------------- #
class AutoGluonAdapter:
    """
    Adapter for AutoGluon TabularPredictor binary classification, compatible with shapiq.

    Returns P(class=1) as a 1D numpy array.
    """

    def __init__(self, predictor, feature_names):
        """
        Parameters:
        -----------
        predictor : TabularPredictor or TimeSeriesPredictor
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
                print_debug(f"Provided: {set(self.feature_names) - set(model_features)}")
                print_debug(f"Missing: {set(model_features) - set(self.feature_names)}")
        except Exception as e:
            print_debug(f"Could not validate features: {e}")

    def __call__(self, X):
        """
        Predict function compatible with shapiq.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Input features of shape (n_samples, n_features)
        
        Returns:
        --------
        np.ndarray
            1D array of probabilities P(class=1) with shape (n_samples,)
        """
        # Convert numpy array to DataFrame with correct feature names
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        elif isinstance(X, pd.DataFrame):
            # Ensure correct column order
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
# Load your trained predictor
predictor = TabularPredictor.load(args.model_path)

train_df = pd.read_csv(args.train_data_path)
test_df = pd.read_csv(args.test_data_path)
X_train = train_df.drop(columns=['slick_tag', 'Time'])
X_test = test_df.drop(columns=['slick_tag', 'Time'])
y_test = test_df["slick_tag"]

assert list(X_train.columns) == list(X_test.columns)
feature_names = X_train.columns.tolist()
print(f"Loaded {len(train_df)} train / {len(test_df)} test samples, {len(feature_names)} features")

# Create the adapter
model_wrapper = AutoGluonAdapter(
    predictor=predictor,
    feature_names=feature_names
)

background = X_train.sample(n=min(200, len(X_train)), random_state=42).values
print_debug(f"Background distribution: {background.shape}")

# Deterministically pick the FULL set of test indices to explain (identical across all jobs
# because of the fixed seed). Each job then processes only its slice of this set, so the
# union of all jobs' outputs is exactly what a single-job run would produce.
if args.sample_size is not None:
    all_to_explain_idx = np.random.default_rng(42).choice(len(X_test), size=args.sample_size, replace=False)
else:
    all_to_explain_idx = np.arange(len(X_test))

# Split the global sample positions into num_jobs roughly equal chunks and pick ours.
global_positions = np.array_split(np.arange(len(all_to_explain_idx)), args.num_jobs)[args.job_index]
to_explain_idx = all_to_explain_idx[global_positions]
samples_to_explain = X_test.iloc[to_explain_idx].values

print(f"Job {args.job_index}/{args.num_jobs}: explaining {len(global_positions)} of {len(all_to_explain_idx)} samples "
      f"(global sample_ids {global_positions[0]}..{global_positions[-1]})")

OUT_DIR = Path(args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

samples_df = X_test.iloc[to_explain_idx].reset_index(drop=True).copy()
samples_df.insert(0, "sample_id", global_positions)
samples_df.insert(1, "original_index", to_explain_idx)
samples_df["y_true"] = y_test.iloc[to_explain_idx].values
samples_df["y_pred_proba"] = model_wrapper(samples_to_explain)

if args.num_jobs == 1:
    samples_csv = OUT_DIR / "explained_samples.csv"
else:
    samples_csv = OUT_DIR / f"explained_samples_job_{args.job_index:03d}.csv"
samples_df.to_csv(samples_csv, index=False)
print(f"Wrote {samples_csv.name} with {len(samples_df)} rows")

# --------------------------------------------------------------------------- #
# Compute k-SII interaction values
# --------------------------------------------------------------------------- #
# Restrict to wind direction features only: 6 features, budget=64 suffices.
# Other features are fixed to each sample's own values so the interaction reflects
# the actual per-sample weather context (e.g. whether wind speed is high or low).
winddir_targets = { # Only for 12h models!
    'winddir_sin_12', 'winddir_sin_11', 'winddir_sin_10',
    'winddir_cos_12', 'winddir_cos_11', 'winddir_cos_10',
}
winddir_mask = np.array([f in winddir_targets for f in feature_names])
winddir_indices = np.where(winddir_mask)[0]
bg_winddir = background[:, winddir_indices]
print(f"Wind direction features ({len(winddir_indices)}): {[feature_names[i] for i in winddir_indices]}")

def make_winddir_model(sample):
    """Freeze `sample`'s non-wind-direction features; return a model that varies only
    the wind-direction sin/cos columns, so shapiq attributes to wind direction alone."""
    def _model(X_wd):
        X_full = np.tile(sample, (len(X_wd), 1))
        X_full[:, winddir_indices] = X_wd
        return model_wrapper(X_full)
    return _model

n_samples = len(samples_to_explain)
for i, sample in enumerate(samples_to_explain):
    global_id = int(global_positions[i])
    print_debug(f"Explaining sample {i + 1}/{n_samples} (global sample_id {global_id})")
    exp = Explainer(
        model=make_winddir_model(sample),
        data=bg_winddir,
        index="k-SII",
        max_order=2,
    )
    v = exp.explain_X(
        sample[winddir_indices].reshape(1, -1),
        budget=64,
        random_state=42
    )
    v[0].to_json_file(OUT_DIR / f"shapley_interactions_sample_{global_id}.json")

print(f"Saved {n_samples} JSONs under {OUT_DIR}/ (job {args.job_index}/{args.num_jobs})")