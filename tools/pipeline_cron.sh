#!/bin/bash
# Run by cron: check raw data for new sessions and submit preprocessing for ready ones (mouse_imaging.pipeline scan).
# Set CONDA_BASE (your conda install) in the crontab line; cron doesn't load your shell setup. See tools/README.md.
#
# Example crontab line (crontab -e), every 10 minutes, never two at once:
# */10 * * * * CONDA_BASE=/molbio/hpc/apps/miniforge3 /usr/bin/flock -n $HOME/.config/mouse_imaging/pipeline.lock $HOME/code/mouse_imaging/tools/pipeline_cron.sh >> $HOME/.config/mouse_imaging/pipeline_cron.log 2>&1
set -o errexit -o nounset -o pipefail
: "${CONDA_BASE:?Set CONDA_BASE to your conda install, e.g. CONDA_BASE=/path/to/miniforge3}"
ENV_NAME="${MOUSE_IMAGING_ENV:-mouse_imaging}"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export CONDA_BASE   # passed on to submitted jobs, so preprocess.slurm can find conda
python -W ignore -m mouse_imaging.pipeline scan 2>&1 | grep -v -e "pynwb not installed" -e "invalid page offset" || true
