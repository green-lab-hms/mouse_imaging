#!/bin/bash
# Cluster settings: edit these for your cluster, or override them when submitting, e.g. sbatch -p <partition> --mem=32G preprocess.slurm ...
#SBATCH -t 1-00:00
#SBATCH -p defq
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -J preprocess
#SBATCH -o preprocess_%j.log
#SBATCH -e preprocess_%j.log

# Usage: sbatch preprocess.slurm <mouse> <date> [session] [steps]
# steps: comma-separated, from suite2p,filters,anndata,qc (default: all four; filters is skipped if there are no filter* subfolders)
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch preprocess.slurm <mouse> <date> [session] [steps]"
    exit 1
fi

mouse=$1
date=$2
session=${3:-session_1}
steps=${4:-suite2p,filters,anndata,qc}

# Uses the conda on your PATH, which batch jobs inherit from the shell you submit from.
# If conda isn't found in the job, set CONDA_BASE to your conda install: sbatch --export=ALL,CONDA_BASE=/path/to/miniforge3 ...
if ! command -v conda > /dev/null 2>&1 && [ -n "$CONDA_BASE" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
eval "$(conda shell.bash hook)"
conda activate mouse_imaging
echo "Running preprocessing steps ${steps} for ${mouse} ${date} ${session}."
python -m mouse_imaging.preprocess --mouse $mouse --date $date --session $session --steps $steps
