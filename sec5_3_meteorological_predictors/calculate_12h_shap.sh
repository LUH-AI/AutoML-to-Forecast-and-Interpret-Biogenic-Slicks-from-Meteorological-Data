#!/bin/bash -l
#SBATCH --job-name=calc_shap
#SBATCH --array=0-99
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a_shap.out
#SBATCH --error=logs/%x_%A_%a_shap.err

# ---------------------------------------------------------------------------
# First-order Shapley values for one trained model.
#
# The array parallelises the TEST SAMPLES of the chosen model into NUM_JOBS
# chunks; it does not iterate over models. Pick the model by uncommenting
# exactly one NAME below, and set the same NAME in plot_shap.sh.
#
#     ARR=$(sbatch --parsable sec5_3_meteorological_predictors/calculate_12h_shap.sh)
#     sbatch --dependency=afterok:$ARR sec5_3_meteorological_predictors/plot_shap.sh
# ---------------------------------------------------------------------------

# --- Setup -----------------------------------------------------------------
cd $SLURM_SUBMIT_DIR
echo "Running array task $SLURM_ARRAY_TASK_ID from $(pwd)"

# --- Activate conda env ------------------------------------------------
CONDA_ENV="${SLICK_CONDA_ENV:-/path/to/.conda/slick}"
module load Miniforge3
conda activate "$CONDA_ENV"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# --- Configuration -----------------------------------------------------
# NAME picks the results/no_static run whose train/test split is used.
NAME="${NAME:-12h_combined_1hour_best}"
#NAME="12h_fino1_2022_1hour_best"
#NAME="12h_spiekeroog_2021_1hour_best"
#NAME="12h_spiekeroog_2024_1hour_best"

# MODEL is the AutogluonModels/ directory to explain, OUT_NAME the output folder.
MODEL="${MODEL:-$NAME}"
OUT_NAME="${OUT_NAME:-$NAME}"

STEM=${NAME%_*}
NUM_JOBS=100   # MUST match the size of --array above

# --- Run -----------------------------------------------------------------
echo "Explaining AutogluonModels/$MODEL on the $NAME split -> results/shapley_values/$OUT_NAME   (chunk $SLURM_ARRAY_TASK_ID / $NUM_JOBS)"

python sec5_3_meteorological_predictors/calc_shap.py \
    -m AutogluonModels/$MODEL \
    -b results/no_static/$NAME/${STEM}_train.csv \
    -d results/no_static/$NAME/${STEM}_test.csv \
    -o results/shapley_values/$OUT_NAME \
    --job_index $SLURM_ARRAY_TASK_ID \
    --num_jobs $NUM_JOBS
