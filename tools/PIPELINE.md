# Automatic preprocessing and the pipeline dashboard

New sessions are preprocessed automatically once their upload has finished. A cron job checks the raw data folders every 10 minutes and submits `preprocess.slurm`, which runs the `suite2p` → `anndata` → `qc` steps, for each session that's ready. The dashboard shows every session's status and QC flags, and has buttons for sessions that aren't processed automatically.

## When is a session processed?

The watcher looks at every `session_N` folder in `<raw_root>/twophoton/<mouse>/<date>/`. A session is **ready** when both of these hold.

**1. All three file types are present:**
- **ScanImage TIFFs:** `<raw_root>/twophoton/<mouse>/<date>/session_N/*.tif`. Stacks in `filter*` subfolders (e.g. `filter2`, `filter1_1024`) are optional, but count toward the upload check below, and the `filters` step processes them.
- **ViRMEn:** `<raw_root>/virmen/<mouse>/<date>/session_N/sessionData.mat`
- **Sync:** `<raw_root>/sync/<mouse>/<date>/session_00N.*`

**2. The upload has finished:**
- **Quiet:** no file has changed for 30 minutes.
- **Stable:** the file list and total size are the same as at the previous scan, 10 minutes earlier.
- **TIFFs look complete:**
  - the TIFF file numbers are contiguous (`_00001`, `_00002`, ...)
  - every TIFF except the last has the same size
  - the last TIFF can be read to its final frame

Ready sessions recorded **on or after the auto-start date** are submitted automatically. That date is the day the pipeline was set up, 2026-10-09. Older sessions are listed as **Not queued** and are processed only when you click **Process**. Up to 2 preprocessing jobs run at a time; the rest wait as **Queued**.

## Session statuses

| Status | Meaning | Button |
|---|---|---|
| ✓ Done | `adata.h5ad` exists | **Reprocess** (overwrites the output) |
| ▶ Running / ◷ Pending | The SLURM job is running or waiting for resources | — |
| • Queued | Ready and waiting for a free job slot | — |
| ↑ Uploading | Files changed recently, or the upload hasn't been stable for two scans | — |
| ! Missing files | One or more of TIFFs, ViRMEn and sync isn't there yet | **Process anyway** if only ViRMEn is missing (preprocesses without behavior) |
| ⚠ TIFF problem | The upload is quiet, but the TIFFs look incomplete, e.g. a missing file number or a truncated last file | **Process anyway** |
| ○ Not queued | Ready, but recorded before the auto-start date | **Process** |
| ✕ Failed | The SLURM job failed or couldn't be submitted | **Retry**; check the job's **Log** link |

**Ignore** (under each session's button) moves a session to the small **Ignored** tab at the right, e.g. a test recording or one you'll never process. Ignored sessions keep updating their status but are never processed automatically, and they don't count in the status tiles. **Restore** brings one back. Running sessions can't be ignored.

The **Job** column estimates how long processing takes: time left and expected finish time for a running job, and total time for other sessions. The estimate comes from the TIFF size and how long finished jobs took per GB, so it improves as more sessions are processed. Two jobs running at once can each take longer.

The **QC** column shows the cell and R+ counts, behavior time and QC flags from `qc_summary.json`, e.g. `z drift`, `R+ cells`, `suite2p log`. Hover over a flag to see the full warning. **QC report ↗** opens the PDF in a new tab.

## Open the dashboard

**Web address: https://w50sdf6l-8050.use.devtunnels.ms**

This works from any browser, without logging into spinoza or starting a VS Code tunnel:

1. Sign in with GitHub as **green-lab-hms**. Other accounts can't open the page.
2. The first time, the page asks for the access token. Get it on spinoza with `cat ~/.config/mouse_imaging/dashboard_token` (jgreen's account). Your browser remembers it for a year.

It refreshes every 30 seconds. **Scan now** checks all sessions immediately instead of waiting for the next cron run.

### How it stays online

It's a Microsoft Dev Tunnel, the same service VS Code tunnels use, and runs as two processes on spinoza:
- `python -m mouse_imaging.dashboard`: the dashboard. It only listens on 127.0.0.1:8050.
- `~/bin/devtunnel host mouse-imaging-dashboard`: forwards the fixed web address to that port. It only lets in GitHub users with access to the tunnel.

A second cron line runs `tools/dashboard_service.sh` every 5 minutes. It restarts either process if it has stopped, for example after spinoza reboots:

```
*/5 * * * * CONDA_BASE=/molbio/hpc/apps/miniforge3 $HOME/code/mouse_imaging/tools/dashboard_service.sh
```

The logs are `~/.config/mouse_imaging/dashboard.log` and `devtunnel.log`. To restart the dashboard after changing `dashboard.py`, stop it with `pkill -f "m mouse_imaging.dashboard"`; cron starts the new code within 5 minutes.

**One-time setup** (already done for jgreen; repeat only if the tunnel is moved to another account):

```bash
curl -sSL -o ~/bin/devtunnel https://aka.ms/TunnelsCliDownload/linux-x64 && chmod +x ~/bin/devtunnel
~/bin/devtunnel user login -g -d             # GitHub device login, as green-lab-hms
~/bin/devtunnel create mouse-imaging-dashboard
~/bin/devtunnel port create mouse-imaging-dashboard -p 8050 --protocol http
```

**If the page stops loading:**
- Check `~/.config/mouse_imaging/devtunnel.log`. If it reports a login or authorization error, run `~/bin/devtunnel user login -g -d` again; the GitHub login expires after a while.
- A tunnel that hasn't been used for 30 days is deleted. Re-create it with the `create` and `port create` commands above; it gets a new web address.

### Without the web address

You can also run your own copy through a VS Code tunnel. Run `python -m mouse_imaging.dashboard --port 8051` in a terminal and open the forwarded port from the **Ports** panel. In a browser-based VS Code (vscode.dev), `localhost` refers to your own computer, so use the **Forwarded Address** shown in that panel, not the printed `localhost` link.

## The cron job

The cron job runs `tools/pipeline_cron.sh` every 10 minutes. It's installed with `crontab -e`:

```
*/10 * * * * CONDA_BASE=/molbio/hpc/apps/miniforge3 /usr/bin/flock -n $HOME/.config/mouse_imaging/pipeline.lock $HOME/code/mouse_imaging/tools/pipeline_cron.sh >> $HOME/.config/mouse_imaging/pipeline_cron.log 2>&1
```

- **What the parts do:** `CONDA_BASE` tells the script where conda is, since cron doesn't load your shell setup. `flock -n` stops a new run from starting while the previous one is still going.
- **Who runs it:** the jobs run as the user who owns the crontab, so only **one** lab member should install it. Anyone can run the dashboard.
- **Checking it:** `tail ~/.config/mouse_imaging/pipeline_cron.log` shows one line per scan with the count of sessions in each status. Each preprocessing job writes `preprocess_<jobid>.log` in the session's derived folder.
- **Pausing it:** comment out the line with `crontab -e`. Comment out the `dashboard_service.sh` line too to take the dashboard offline, then stop both processes: `pkill -f "m mouse_imaging.dashboard"; pkill -f "devtunnel host"`.

## From the command line

```bash
python -m mouse_imaging.pipeline status                          # table of every session's status
python -m mouse_imaging.pipeline scan --no-submit                # update the status without submitting anything
python -m mouse_imaging.pipeline scan                            # what cron runs: update and submit ready sessions
python -m mouse_imaging.pipeline queue JG6 260929 session_2      # same as the Process button
python -m mouse_imaging.pipeline queue JG6 261005 session_1 --force   # Process anyway (no ViRMEn file, or a TIFF problem)
python -m mouse_imaging.pipeline ignore JG1 260529 session_1     # same as the Ignore button; `unignore` restores it
```

## Settings

The status is kept in `<derived_root>/.pipeline/state.json`, shared by the cron job and the dashboard. Settings go in the `[pipeline]` section of your config file (see [Configuration](../README.md#configuration)); all are optional:

```toml
[pipeline]
auto_start_date = "261009"     # sessions recorded on or after this date (YYMMDD) are processed automatically
quiet_minutes = 30             # how long files must be unchanged before an upload counts as finished
max_concurrent_jobs = 2        # preprocessing jobs at a time (each uses 16 CPUs and 64 GB)
state_dir = "/path/to/state"   # default: <derived_root>/.pipeline
```

## Troubleshooting

- **A session stays "Uploading":** files are still changing, or the folder was copied in several passes. The status turns ready about 30–40 minutes after the last change.
- **"TIFF problem: The last TIFF cannot be read":** the last file ends mid-frame. This happens if the copy was interrupted, so re-copy it if possible. If the acquisition itself was cut short, **Process anyway**.
- **A job failed:** open its **Log** link. After fixing the cause, click **Retry**.
- **Nothing is being submitted:** check `~/.config/mouse_imaging/pipeline_cron.log` for errors, and that the cron line exists with `crontab -l`.
