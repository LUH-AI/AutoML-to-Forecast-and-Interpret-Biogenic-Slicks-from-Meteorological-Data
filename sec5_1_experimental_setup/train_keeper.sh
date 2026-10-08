#!/bin/bash -l
# ---------------------------------------------------------------------------
# Train ONE model and keep it, for the Shapley analysis.
#
# The training grid in run_array.sh passes --cleanup and deletes every
# model it fits, so nothing is left to explain. This trains a single model
# without --cleanup and saves it under AutogluonModels/$NAME, which is where
# sec5_3_meteorological_predictors/calculate_12h_shap.sh and calculate_12h_shapiq.sh look for it.
#
# It deliberately passes no -o. The windowed dataset and the 70/30 split are
# identical for every seed (the split is fixed at random_state=42), so the
# array already writes results/no_static/$NAME/${STEM}_{train,test}.csv, and
# writing them a second time from here would only risk two jobs writing the
# same file at once.
#
# Edit NAME/DATA/WINDOW below, then:
#     sbatch sec5_1_experimental_setup/train_keeper.sh
# Run one at a time, or give each its own NAME.
# ---------------------------------------------------------------------------

#SBATCH --job-name=slick_keeper
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

NAME="${NAME:-12h_combined_1hour_best}"
DATA="${DATA:-build/1hour_combined_dataset.csv}"
WINDOW="${WINDOW:-12}"
SEED="${SEED:-0}"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
echo "[$(date '+%F %T')] keeper $NAME on $(hostname), cwd=$(pwd)"

if [[ -e "AutogluonModels/$NAME" ]]; then
    echo "ERROR: AutogluonModels/$NAME already exists - move it aside first" >&2
    exit 1
fi

# --- Activate conda env --------------------------------------------------
CONDA_ENV="${SLICK_CONDA_ENV:-/path/to/.conda/slick}"
module load Miniforge3
conda activate "$CONDA_ENV"
echo "Loaded conda env: $(which python)"

# --- Start Ray cluster (head + worker on the 2 allocated nodes) ----------
head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
# 6300 sits below the range run_array.sh uses (6400 + task_id * 10), so a
# keeper and the training array can run at the same time.
port=6300
dashboard_port=$((port + 1))
ray_tmpdir="/tmp/ray_keeper_${SLURM_JOB_ID}"

unset SLURM_TRES_PER_TASK

echo "STARTING RAY HEAD at $head_node ($head_node_ip)"
srun --nodes=1 --ntasks=1 -w "$head_node" \
     "$SLURM_SUBMIT_DIR/sec5_1_experimental_setup/start-head.sh" "$head_node_ip" "$port" "$dashboard_port" "$ray_tmpdir" &
sleep 10

worker_num=$(( SLURM_JOB_NUM_NODES - 1 ))
echo "STARTING $worker_num RAY WORKER(s)"
srun -n "$worker_num" --nodes="$worker_num" --ntasks-per-node=1 \
     --exclude "$head_node" \
     "$SLURM_SUBMIT_DIR/sec5_1_experimental_setup/start-worker.sh" "$head_node_ip:$port" "$ray_tmpdir" &
sleep 5

# --- Train and keep ------------------------------------------------------
echo "----------------------------------------------------------------"
echo "Training $NAME  (-w $WINDOW, seed $SEED) -> AutogluonModels/$NAME"
echo "----------------------------------------------------------------"

python sec5_1_experimental_setup/train_tabular_predictor.py \
    -d "$DATA" \
    -w "$WINDOW" \
    -p best \
    --seed "$SEED" \
    --model_dir "AutogluonModels/$NAME" \
    --debug
RC=$?

echo "[$(date '+%F %T')] keeper exited with code $RC"
[[ $RC -eq 0 ]] && echo "Model saved at AutogluonModels/$NAME"
exit $RC
