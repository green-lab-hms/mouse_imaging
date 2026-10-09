#!/bin/bash
# Start a VS Code tunnel on a compute node, so you can work on the cluster from VS Code on your own computer
# or from https://vscode.dev in a browser. Step-by-step guide: tools/README.md.
#
# Usage: sbatch vscode.sh [tunnel_name]      (tunnel_name defaults to spinozatunnel, max 20 characters)
# Then read ~/vscode.out for the GitHub login code and the tunnel link.
#
# Resources for everything you run inside VS Code (terminals, notebooks): edit these, or override when submitting,
# e.g. sbatch --mem=32g -c 4 vscode.sh
#SBATCH -p defq                   # Partition
#SBATCH --mem=4g                  # RAM
#SBATCH --time=12:00:00           # Wall time; the tunnel closes when the job ends
#SBATCH -c 1                      # CPU cores
#SBATCH -J vscode
#SBATCH -o /dev/null              # Output goes to ~/vscode.out below, wherever you submit from

set -o errexit -o nounset -o pipefail
exec > "$HOME/vscode.out" 2>&1

TUNNEL_NAME="${1:-spinozatunnel}"
VSCODE_DIR="$HOME/code/vscode" # where the VS Code command-line tool is installed
mkdir -p "$VSCODE_DIR"
echo "$VSCODE_DIR"

# Download the VS Code command-line tool the first time
if [ -e "$VSCODE_DIR/code" ]; then
    echo "using $VSCODE_DIR/code"
else
    echo "install a new version of code"
    curl -L 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' | tar -C "$VSCODE_DIR" -xzf -
fi

# Log in with GitHub: prints a code to enter at https://github.com/login/device
#"$VSCODE_DIR/code" tunnel user login --provider microsoft
"$VSCODE_DIR/code" tunnel user login --provider github

# Accept the license terms and start the tunnel
"$VSCODE_DIR/code" tunnel --accept-server-license-terms --name "$TUNNEL_NAME"
