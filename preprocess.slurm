#!/bin/bash
#SBATCH -t 1-00:00
#SBATCH -p defq
#SBATCH -J preprocess
#SBATCH -o preprocess_%j.log
#SBATCH -e preprocess_%j.log
#SBATCH -c 16
#SBATCH --mem=64G

# Usage: sbatch preprocess.slurm <mouse> <date> [session]
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch preprocess.slurm <mouse> <date> [session]"
    exit 1
fi

mouse=$1
date=$2
session=${3:-session_1}

source /molbio/hpc/apps/miniforge3/etc/profile.d/conda.sh
conda activate mouse_imaging
echo "Running suite2p preprocessing for ${mouse} ${date} ${session}."
python ~/code/mouse_imaging/preprocess.py --mouse $mouse --date $date --session $session
