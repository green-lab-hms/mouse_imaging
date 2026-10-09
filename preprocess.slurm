#!/bin/bash
#SBATCH -t 1-00:00
#SBATCH -p defq
#SBATCH -J preprocess
#SBATCH -o preprocess_%j.log
#SBATCH -e preprocess_%j.log
#SBATCH -c 16
#SBATCH --mem=64G

# Usage: sbatch preprocess.slurm <mouse> <date> [session] [steps]
# steps: comma-separated, from suite2p,anndata (default: both)
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch preprocess.slurm <mouse> <date> [session] [steps]"
    exit 1
fi

mouse=$1
date=$2
session=${3:-session_1}
steps=${4:-suite2p,anndata}

source /molbio/hpc/apps/miniforge3/etc/profile.d/conda.sh
conda activate mouse_imaging
echo "Running preprocessing steps ${steps} for ${mouse} ${date} ${session}."
python -m mouse_imaging.preprocess --mouse $mouse --date $date --session $session --steps $steps
