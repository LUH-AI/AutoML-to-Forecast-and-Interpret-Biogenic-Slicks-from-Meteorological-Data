#!/bin/bash -l
#SBATCH --job-name=plot_shap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j_plot.out
#SBATCH --error=logs/%x_%j_plot.err

# ---------------------------------------------------------------------------
# Plot the Shapley values produced by calculate_12h_shap.sh.
# Submit AFTER the array job finishes:
#     ARR=$(sbatch --parsable sec5_3_meteorological_predictors/calculate_12h_shap.sh)
#     sbatch --dependency=afterok:$ARR sec5_3_meteorological_predictors/plot_shap.sh
# ---------------------------------------------------------------------------

# --- Setup -----------------------------------------------------------------
cd $SLURM_SUBMIT_DIR

# --- Activate conda env ------------------------------------------------
CONDA_ENV="${SLICK_CONDA_ENV:-/path/to/.conda/slick}"
module load Miniforge3
conda activate "$CONDA_ENV"

# --- Configuration -----------------------------------------------------
# MUST match NAME in calculate_12h_shap.sh, otherwise this reads a directory
# the array never wrote to.
# Must match OUT_NAME of the calculate job that produced the values.
OUT_NAME="${OUT_NAME:-${NAME:-12h_combined_1hour_best}}"

# --- Run -----------------------------------------------------------------
echo "Plotting SHAP results for: $OUT_NAME"

python sec5_3_meteorological_predictors/plot_shap.py \
    -o results/shapley_values/$OUT_NAME
