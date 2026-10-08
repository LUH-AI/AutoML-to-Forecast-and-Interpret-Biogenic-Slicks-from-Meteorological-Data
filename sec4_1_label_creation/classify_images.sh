#!/bin/bash -l
#SBATCH --job-name=classify_images
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# Stage 2 of the labelling chain: apply the trained image classifier to the 
# images listed in to_classify.csv.
#
# MODEL is the AutoGluon checkpoint from train_image_model.py.
# IMAGES is a folder holding those images.
# predictions.csv is written to the working directory, so run one site at a
# time or move the result afterwards.
# ---------------------------------------------------------------------------

# --- Setup -----------------------------------------------------------------
cd $SLURM_SUBMIT_DIR
echo "Running job from $(pwd)"

# --- Activate conda env ------------------------------------------------
CONDA_ENV="${SLICK_CONDA_ENV:-/path/to/.conda/slick}"
module load Miniforge3
conda activate "$CONDA_ENV"

# --- Configuration -----------------------------------------------------
MODEL="AutogluonModels/{MODEL_NAME}"
IMAGES="{PATH_TO/IMAGE_FOLDER}"

# --- Run -----------------------------------------------------------------
echo "Classifying $IMAGES with $MODEL"
srun python sec4_1_label_creation/classify_images.py -m "$MODEL" -d "$IMAGES"
