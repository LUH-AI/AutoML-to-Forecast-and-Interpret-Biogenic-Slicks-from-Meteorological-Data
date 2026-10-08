#!/bin/bash -l
# ---------------------------------------------------------------------------
# SLURM array job - runs every line in sec5_1_experimental_setup/jobs.txt as its own task.
# Each array task starts its own 2-node Ray cluster (one head + one worker)
# before invoking AutoGluon.
#
# Generate the job list first, then launch from the project root:
#     python sec5_1_experimental_setup/generate_jobs.py
#     sbatch sec5_1_experimental_setup/run_array.sh
#
# --array below MUST match the number of lines generate_jobs.py reports.
# The full experiment grid is 4 datasets x 10 window lengths x 5 seeds = 200.
# Resources per job: 2 nodes x 8 CPUs x 32G = 16 CPUs and 64 GB in total (--mem is per node).
# ---------------------------------------------------------------------------

#SBATCH --job-name=slick_train
#SBATCH --array=1-200
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
echo "[$(date '+%F %T')] array task $SLURM_ARRAY_TASK_ID on $(hostname), cwd=$(pwd)"

# --- Activate conda env --------------------------------------------------
CONDA_ENV="${SLICK_CONDA_ENV:-/path/to/.conda/slick}"
module load Miniforge3
conda activate "$CONDA_ENV"
echo "Loaded conda env: $(which python)"

# --- Start Ray cluster (head + worker on the 2 allocated nodes) ----------
head_node=$(hostname)
head_node_ip=$(hostname --ip-address)
# Per-task port and temp dir, so concurrent array tasks on one node cannot collide.
port=$((6400 + SLURM_ARRAY_TASK_ID * 10))
dashboard_port=$((port + 1))
ray_tmpdir="/tmp/ray_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

# Fix SLURM_CPUS_PER_TASK and SLURM_TRES_PER_TASK conflict, that caused
# "cpus-per-task set by two different environment variables" fatal.
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

# --- Pick the command for this array task --------------------------------
JOBS_FILE="sec5_1_experimental_setup/jobs.txt"
if [[ ! -f "$JOBS_FILE" ]]; then
    echo "ERROR: $JOBS_FILE not found (run python sec5_1_experimental_setup/generate_jobs.py)" >&2
    exit 1
fi

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS_FILE")
if [[ -z "$CMD" ]]; then
    echo "ERROR: line $SLURM_ARRAY_TASK_ID of $JOBS_FILE is empty" >&2
    exit 1
fi

echo "----------------------------------------------------------------"
echo "CMD: $CMD"
echo "----------------------------------------------------------------"

# --- Run -----------------------------------------------------------------
eval "$CMD"
RC=$?

echo "----------------------------------------------------------------"
echo "[$(date '+%F %T')] array task $SLURM_ARRAY_TASK_ID exited with code $RC"
exit $RC
