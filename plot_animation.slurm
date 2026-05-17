#!/bin/bash
#SBATCH -t 0-1:00
#SBATCH -p short
#SBATCH -J animation
#SBATCH -o animation_%j.log
#SBATCH -e animation_%j.log
#SBATCH -c 1
#SBATCH --mem=16G

mouse=$1
date=$2
session=$3
trial=$4
path_out=$5
fps=30

source /n/app/conda2/etc/profile.d/conda.sh
conda activate mouse_imaging
echo "Rendering animation for ${mouse} ${date} ${session} trial ${trial} at ${fps} frames per second. Saving in: \n${path_out}."
python ~/code/mouse_imaging/plot_animation.py --mouse $mouse --date $date --session $session --trial $trial --path_out $path_out --fps $fps

