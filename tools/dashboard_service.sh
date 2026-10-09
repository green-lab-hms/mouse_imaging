#!/bin/bash
# Keep the pipeline dashboard and its dev tunnel running. Run by cron every 5 minutes; each process holds a lock,
# so a copy that's already running is left alone and a crashed one is restarted. See tools/PIPELINE.md.
#   CONDA_BASE   conda install (cron doesn't load your shell setup)
#   TUNNEL_ID    dev tunnel created with `devtunnel create` (default: mouse-imaging-dashboard)
set -u
CONF=$HOME/.config/mouse_imaging
TUNNEL_ID=${TUNNEL_ID:-mouse-imaging-dashboard}
PORT=${PORT:-8050}
DEVTUNNEL=${DEVTUNNEL:-$HOME/bin/devtunnel}

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate mouse_imaging
export CONDA_BASE  # used by the dashboard's Process buttons to submit jobs

URL=$("$DEVTUNNEL" show "$TUNNEL_ID" 2>/dev/null | grep -o "https://[^ ]*-$PORT\.[^ ]*" | head -1)
nohup flock -n "$CONF/dashboard.lock" python -W ignore -m mouse_imaging.dashboard --port "$PORT" ${URL:+--url "${URL%/}"} \
    >> "$CONF/dashboard.log" 2>&1 < /dev/null &
nohup flock -n "$CONF/devtunnel.lock" "$DEVTUNNEL" host "$TUNNEL_ID" >> "$CONF/devtunnel.log" 2>&1 < /dev/null &
