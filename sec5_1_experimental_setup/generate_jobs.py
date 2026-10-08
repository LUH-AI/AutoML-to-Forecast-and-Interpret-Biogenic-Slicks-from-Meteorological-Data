"""Generate jobs.txt from the training plan.

Run this from the project root:
    python sec5_1_experimental_setup/generate_jobs.py

Writes sec5_1_experimental_setup/jobs.txt with one full python command per line.
The SLURM array script (sec5_1_experimental_setup/run_array.sh) picks line N for array task N.

The single place the experiment grid is defined (datasets, window lengths, seeds); jobs.txt
is generated and untracked, so the grid can only be changed here.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------
# Folder name : {input_label}_{site}_{sample_rate}_{quality}
# CSV name    : {input_label}_{site}_{sample_rate}.csv
# Full path   : results/no_static/<folder>/<csv>
# ---------------------------------------------------------------------------

# Paths are relative to the project root (also the SLURM submit directory), so generated
# commands run the same locally and on the cluster. Datasets are expected at build/;
# set $SLICK_DATA_DIR if they live somewhere else.
DATA_DIR = os.environ.get("SLICK_DATA_DIR", "build")

DATA_FILES = {
    "fino1_2022":  f"{DATA_DIR}/1hour_fino1_2022.csv",
    "spiekeroog_2024": f"{DATA_DIR}/1hour_spiekeroog_2024.csv",
    "spiekeroog_2021": f"{DATA_DIR}/1hour_spiekeroog_2021.csv",
    "combined":       f"{DATA_DIR}/1hour_combined_dataset.csv",
}

PRESET       = "best"
SCRIPT       = "sec5_1_experimental_setup/train_tabular_predictor.py"
RESULTS_ROOT = "results/no_static"
SEEDS        = [0, 1, 2, 3, 4]  # 5 runs per model to measure variance

RUNS = [
    ("1h",     1),
    ("3h",     3),
    ("6h",     6),
    ("12h",    12),
    ("24h",    24),
    ("48h",    48),
    ("4days",  96),   
    ("1week",  168),
    ("2weeks", 336),  
    ("4weeks", 672),
]

GROUPS = [
    ("fino1_2022", "1hour", RUNS),
    ("spiekeroog_2021", "1hour", RUNS),
    ("spiekeroog_2024", "1hour", RUNS),
    ("combined", "1hour", RUNS),
]


def build_command(site: str, sample_rate: str, input_label: str, window_size: int, seed: int) -> str:
    """Build one train_tabular_predictor.py invocation, following the folder/CSV naming
    convention in the "Conventions" comment above, so results/, sec5_2_slick_prediction/ and
    sec5_3_meteorological_predictors/ can all parse a run's site/window/seed back out of its path."""
    folder     = f"{input_label}_{site}_{sample_rate}_{PRESET}"
    csv        = f"{input_label}_{site}_{sample_rate}.csv"
    output_dir = f"{RESULTS_ROOT}/{folder}"
    output     = f"{output_dir}/{csv}"
    # mkdir -p makes each command self-contained — runnable via sbatch or manually.
    # Train/test CSVs are written to the shared folder (same split every seed).
    # Predictions are saved as {base}_seed{N}_predictions.csv inside the same folder.
    return (
        f'mkdir -p "{output_dir}" && '
        f'python {SCRIPT} '
        f'-d "{DATA_FILES[site]}" '
        f'-w {window_size} '
        f'-p "{PRESET}" '
        f'-o "{output}" '
        f'--seed {seed} '
        f'--cleanup --debug'
    )



def main() -> None:
    """Expand GROUPS x SEEDS into one command per run and write them to jobs.txt."""
    commands = []
    for site, sample_rate, runs in GROUPS:
        for input_label, window_size in runs:
            for seed in SEEDS:
                commands.append(build_command(site, sample_rate, input_label, window_size, seed))

    # A wrong data path costs one failed array task per job, so say so here instead.
    missing = [p for p in DATA_FILES.values() if not Path(p).is_file()]
    if missing:
        print(f"WARNING: {len(missing)} of {len(DATA_FILES)} datasets not found "
              f"relative to {Path.cwd()}:")
        for path in missing:
            print(f"  {path}")
        print("Run this from the project root, or set $SLICK_DATA_DIR.")

    out_path = Path(__file__).parent / "jobs.txt"
    out_path.write_text("\n".join(commands) + "\n")
    print(f"Wrote {len(commands)} jobs to {out_path}")
    print(f"Update --array=1-{len(commands)} in sec5_1_experimental_setup/run_array.sh if the count changed.")


if __name__ == "__main__":
    main()
