#!/bin/bash
# Cluster settings: edit these for your cluster, or override them when submitting, e.g. sbatch -p <partition> --mem=32G preprocess.slurm ...
#SBATCH -t 1-00:00
#SBATCH -p defq
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -J preprocess
#SBATCH -o preprocess_%j.log
#SBATCH -e preprocess_%j.log

# Usage: sbatch preprocess.slurm <mouse> <date> [session] [steps]
# steps: comma-separated, from suite2p,filters,anndata,qc,compress (default: all; filters is skipped if there are no filter* subfolders)
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch preprocess.slurm <mouse> <date> [session] [steps]"
    exit 1
fi

mouse=$1
date=$2
session=${3:-session_1}
steps=${4:-suite2p,filters,anndata,qc,compress}

# Uses the conda on your PATH, which batch jobs inherit from the shell you submit from.
# If conda isn't found in the job, set CONDA_BASE to your conda install: sbatch --export=ALL,CONDA_BASE=/path/to/miniforge3 ...
if ! command -v conda > /dev/null 2>&1 && [ -n "$CONDA_BASE" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
eval "$(conda shell.bash hook)"
conda activate mouse_imaging
# spinoza's SLURM doesn't confine jobs to their cores (ConstrainCores=no), and torch/numba/BLAS otherwise start a
# thread per core of the whole node, so cap them at this job's CPUs
threads=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads NUMBA_NUM_THREADS=$threads

echo "Running preprocessing steps ${steps} for ${mouse} ${date} ${session} on ${threads} threads."
python -m mouse_imaging.preprocess --mouse $mouse --date $date --session $session --steps $steps
