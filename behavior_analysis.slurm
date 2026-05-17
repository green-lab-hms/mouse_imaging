#!/bin/bash
#SBATCH -t 0-0:30
#SBATCH -p short 
#SBATCH -J beh_analysis
#SBATCH -o behavior_analysis_%j.log
#SBATCH -e behavior_analysis_%j.log
#SBATCH -c 1
#SBATCH --mem=16G

mouse=$1

source /n/app/conda2/etc/profile.d/conda.sh
conda activate mouse_imaging
echo "Compiling behavior analysis for ${mouse}."
python ~/code/mouse_imaging/behavior_analysis.py --mouse $mouse

