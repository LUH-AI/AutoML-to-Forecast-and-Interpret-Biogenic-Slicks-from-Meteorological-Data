#!/bin/bash -l
#SBATCH --job-name=calc_shapiq
#SBATCH --array=0-49
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a_shapiq.out
#SBATCH --error=logs/%x_%A_%a_shapiq.err

# ---------------------------------------------------------------------------
# Second-order Shapley interaction values, used to recombine the wind-direction
# sine/cosine pair into a single directional contribution.
#
# As in calculate_12h_shap.sh, the array parallelises the TEST SAMPLES of the
# chosen model into NUM_JOBS chunks, NOT the models. Uncomment exactly one NAME.
#
# Afterwards, stitch the per-chunk outputs back together. plot_wind_direction.py
# looks for explained_samples.csv INSIDE the output directory, so run this there:
#     cd results/shapley_interactions/<NAME>
#     awk 'NR==1 || FNR>1' explained_samples_job_*.csv > explained_samples.csv
# ---------------------------------------------------------------------------

# --- Setup -----------------------------------------------------------------
cd $SLURM_SUBMIT_DIR
echo "Running array task $SLURM_ARRAY_TASK_ID from $(pwd)"

# --- Activate conda env ------------------------------------------------
CONDA_ENV="${SLICK_CONDA_ENV:-/path/to/.conda/slick}"
module load Miniforge3
conda activate "$CONDA_ENV"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# --- Configuration -----------------------------------------------------
# NAME picks the results/no_static run whose train/test split is used.
NAME="${NAME:-12h_combined_1hour_best}"
# MODEL is the AutogluonModels/ directory to explain, OUT_NAME the output folder.
MODEL="${MODEL:-$NAME}"
OUT_NAME="${OUT_NAME:-$NAME}"

STEM=${NAME%_*}
NUM_JOBS=50   # MUST match the size of --array above

# --- Run -----------------------------------------------------------------
echo "Explaining AutogluonModels/$MODEL on the $NAME split -> results/shapley_interactions/$OUT_NAME   (chunk $SLURM_ARRAY_TASK_ID / $NUM_JOBS)"
echo "Executing sec5_3_meteorological_predictors/calc_shap_interactions.py"

python sec5_3_meteorological_predictors/calc_shap_interactions.py \
    -m AutogluonModels/$MODEL \
    -b results/no_static/$NAME/${STEM}_train.csv \
    -d results/no_static/$NAME/${STEM}_test.csv \
    -o results/shapley_interactions/$OUT_NAME \
    --job_index $SLURM_ARRAY_TASK_ID \
    --num_jobs $NUM_JOBS
