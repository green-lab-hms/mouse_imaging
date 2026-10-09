# Working on spinoza with VS Code

[`vscode.sh`](vscode.sh) is a SLURM job that starts a [VS Code tunnel](https://code.visualstudio.com/docs/remote/tunnels) on a compute node. Through the tunnel you can edit code, use terminals and run notebooks on spinoza from VS Code on your own computer, without SSH port forwarding. Everything you run in VS Code runs inside this job, on the compute node.

These steps go from nothing to running the example notebook. Steps 1 and 2 are only needed once.

## Step 1: set up VS Code on your computer (once)

1. Install [VS Code](https://code.visualstudio.com).
2. In VS Code, open the Extensions view (Ctrl/Cmd+Shift+X) and install **Remote - Tunnels** (`ms-vscode.remote-server`).
3. Have a GitHub account ready. The tunnel uses it so that only you can connect.

## Step 2: set up the Python environment on spinoza (once)

Log in to spinoza with SSH, then install the package and register its notebook kernel. See [Installation](../README.md#installation) for details.

```bash
git clone https://github.com/green-lab-hms/mouse_imaging.git ~/code/mouse_imaging
cd ~/code/mouse_imaging
conda env create -f environment.yml
conda activate mouse_imaging
python -m ipykernel install --user --name mouse_imaging --display-name "Python (mouse_imaging)"
```

## Step 3: start the tunnel

On spinoza, submit the tunnel job and wait for it to start:

```bash
sbatch --mem=32g -c 4 ~/code/mouse_imaging/tools/vscode.sh
squeue -u $USER          # wait until the vscode job shows ST = R
```

- **Resources:** the job's defaults are 1 CPU, 4 GB of RAM and 12 hours. `--mem=32g -c 4` gives enough memory to load a session in a notebook; adjust as needed.
- **Tunnel name:** an optional argument after the script path sets it (default `spinozatunnel`), e.g. `... vscode.sh my-tunnel`. Use a different name if you share a GitHub account with someone.

## Step 4: log in with GitHub

1. Read the job's log, which is always written to `~/vscode.out`:
   ```bash
   cat ~/vscode.out
   ```
   Within a few seconds of the job starting, it shows a login code:
   ```
   To grant access to the server, please log into https://github.com/login/device and use code ABCD-1234
   ```
2. Open https://github.com/login/device in a browser, enter the code and authorize **Visual Studio Code**.
3. Run `cat ~/vscode.out` again. When the tunnel is ready it shows:
   ```
   ➜  Tunnel:   spinozatunnel
   ➜  Open:  https://vscode.dev/tunnel/spinozatunnel/...
   ```

The first run also downloads the VS Code command-line tool into `~/code/vscode/`, which takes a minute.

## Step 5: connect from VS Code

1. In VS Code on your computer, open the Command Palette (Ctrl/Cmd+Shift+P) and run **Remote-Tunnels: Connect to Tunnel...**.
2. Sign in with the **same GitHub account** as in step 4 when asked, then choose your tunnel (`spinozatunnel`).
3. A new window opens. The bottom-left corner shows **Tunnel: spinozatunnel** once it's connected. Next time, you can also connect from the **Remote Explorer** sidebar.

Without installing VS Code, you can open the `Open:` link from `~/vscode.out` in a browser instead. The remaining steps are the same.

## Step 6: open the repository

1. **File → Open Folder...**, enter `~/code/mouse_imaging` (or the full path, e.g. `/home/green_lab/<user>/code/mouse_imaging`) and click OK.
2. If asked, trust the folder's authors.
3. Open the Extensions view and install **Python** and **Jupyter** on the tunnel. VS Code shows an **Install in Tunnel** button for extensions that aren't installed on the remote side yet. This is needed once per tunnel name.

## Step 7: run the example notebook

1. In the Explorer, open `examples/example_session.ipynb`.
2. Click **Select Kernel** in the top right of the notebook. Choose **Jupyter Kernel...** → **Python (mouse_imaging)**, or **Python Environments...** → **mouse_imaging**.
3. Run the cells one at a time with Shift+Enter, or all at once with **Run All** at the top. The notebook loads JG6/260929/session_1 from the shared derived data folder, so you need read access to it (see [Configuration](../README.md#configuration)).
4. To analyze your own sessions, copy the notebook, e.g. right-click → **Copy**, then paste and rename it, and change the mouse and date in the loading cell.

## Step 8: finish

- **Closing VS Code:** closing the window doesn't stop the job. You can reconnect (step 5) any time until the job's time limit.
- **Stopping the tunnel:** stop it when you're done, so it doesn't hold resources:
  ```bash
  squeue -u $USER          # find the vscode job's ID
  scancel <jobid>
  ```
- **Next session:** start again at step 3. You'll need to log in with GitHub again (step 4) each time.

## Tips and troubleshooting

- **Heavy jobs:** run heavy work, such as `preprocess.slurm`, as its own SLURM job rather than inside the tunnel. You can submit it from a terminal in VS Code (Terminal → New Terminal).
- **No login code in `~/vscode.out`:** check that the job is running with `squeue -u $USER`. If it's still pending (`ST = PD`), wait for resources.
- **"Tunnel name already in use":** only one tunnel per name can run at a time. Cancel the old vscode job, or use another name.
- **The kernel isn't listed:** repeat the `ipykernel install` command in step 2, then reload the window (Command Palette → **Developer: Reload Window**).
- **The kernel dies while loading data:** the job ran out of memory. Restart the tunnel with more, e.g. `--mem=64g`.
- **Disconnected after 12 hours:** the job reached its time limit. Resubmit, optionally with a longer `--time=24:00:00`.
