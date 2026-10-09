# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`mouse_imaging` is a Python package for analyzing two-photon calcium imaging of mice running virtual-reality (ViRMEn) tasks, plus optional holographic photostimulation. It turns raw ScanImage TIFFs, ViRMEn behavior files and DAQ sync files into one `AnnData` object per session, which the analysis and plotting modules then use.

The code was originally written for the HMS O2 cluster (`/n/data2/...`, `/n/scratch/...`) and is being migrated to the `spinoza` cluster. Expect leftover O2 paths and code that depends on files the new layout doesn't produce yet (see "Migration state" below).

## Environment and running

- Conda env: `mouse_imaging`, at `~/.conda/envs/mouse_imaging`. It uses suite2p 1.1 and torch. `~/code` is on the env's path through a `.pth` file, so the package imports as `mouse_imaging` from anywhere. The repo directory itself has to be named `mouse_imaging`.
- Activate on spinoza: `source /molbio/hpc/apps/miniforge3/etc/profile.d/conda.sh && conda activate mouse_imaging`
- Cluster: SLURM, a single node `spinoza` (partition `defq`, 96 CPUs, about 250 GB RAM, **no GPU**, so suite2p/cellpose run with `torch_device='cpu'`).
- There are no tests, linter or build step. To check your changes, import the module and run the relevant script:
  ```bash
  cd ~/code && python -c "import mouse_imaging.preprocess"
  ```

### Entry points

| Command | What it does |
|---|---|
| `sbatch preprocess.slurm <mouse> <date> [session]` | Runs suite2p on all planes of a session together (`preprocess.py`). `session` defaults to `session_1`. The log goes to `preprocess_<jobid>.log` in the directory you submit from. |
| `python session.py --mouse M --date D [--session S] [--ops default_ops]` | Builds the `Session` and `AnnData` from preprocessed outputs and saves `adata.h5ad`. |
| `python session.py --mouse M --date D --update` | Re-runs `update_function` (photostim influence, tuning, regressions) on an existing `adata.h5ad`. |
| `sbatch plot_animation.slurm ...` | Renders VR and activity movies (`plot_animation.py`). This file still uses the O2 conda path. |

## Data layout

Roots are set in `options.default_ops()`; `session.define_path()` builds every file path from them:

- Raw: `/data/green_lab/shared/data/raw/{twophoton,virmen,sync}/<mouse>/<date>/<session>/`
- Derived: `/data/green_lab/shared/data/derived/twophoton/<mouse>/<date>/<session>/`, containing `suite2p/plane0..N/`, `adata.h5ad`, `session.pickle` and `metadata.pickle`

Sessions are identified by a key dict, `{'mouse', 'date', 'session'}`. Dates are `YYMMDD` strings and sessions are `session_N`. Raw TIFF filenames encode acquisition info, for example `920nm_G1R1_V1_L123_00001_00001.tif` gives wavelength, emission filters (color+number pairs), region and depth. `parse_si_filename` parses these names.

## Architecture

**Configuration ("ops").** `options.py` defines `default_ops()`, a plain dict holding paths, behavior constants, sync column mapping, ViRMEn column names and `ops['suite2p_settings']` (a suite2p 1.x `default_settings()` dict with overrides). Variants are selected by function name: scripts take `--ops <name>` and call `getattr(options, name)()`. `options.py` also holds cell-type calling and post-processing functions (`call_celltypes_*`, `process_*`), and `session.main` finds these through `ops['process_fcn']`. `suite2p` and `torch` are imported inside `default_ops` on purpose, so analysis-only environments don't need them.

**Metadata.** `session.get_metadata(path)` reads the ScanImage header of the first raw TIFF (`parse_si_metadata` flattens it into `'SI.hX.y'` keys) and exposes `nslices`, `nflyback`, `nchannels`, `Ly`, `Lx` and `volume_rate`. Multi-element SI values such as `[1;2]` stay strings.

**Preprocessing (`preprocess.py`).** This passes all planes to suite2p in a single `run_s2p(settings=..., db=...)` call. Flyback frames count toward `nplanes` and are skipped with `ignore_flyback`. suite2p's plane folders are **0-based**.

**Session assembly (`session.Session`).** The constructor runs the whole pipeline:
1. `Sync` loads the DAQ file (`.abf` or `.h5`), renames columns to canonical names (`Virmen`, `ScanImage`, `Ball_*`, `Licks`) and finds VR-frame and imaging-volume indices from the clock edges.
2. Loads the ViRMEn `sessionData.mat` (`load_vr`) and aligns it to sync.
3. Loads suite2p `stat` and activity per plane. Columns are named `plane{p}_source{i}`; `X` and the `dcnv` layer hold OASIS deconvolved activity, and the `dF` layer holds dF/F. Then it crops imaging and VR to their overlap.
4. Builds `obs` (per imaging frame: VR variables resampled to the frame times) and `var` (per source: channel intensities from mean-image hyperstacks, cellpose overlap).
5. Optionally runs photostim processing (`photostimulation.process_photostim_session`).
6. Runs the maze-specific variables in `importlib.import_module('mouse_imaging.' + ops['env']).main(self, ...)`. `env` is a module name such as `tmaze` or `wideLinearTrack`.
7. Pickles itself. `load_as_anndata` wraps the result as `AnnData(X, obs, var, uns, layers)`, and `uns` carries `path`, `metadata` and `ops`.

**Downstream.** `analysis.py` (binning, triggered averages `trigger_*`, tuning, regression), `photostimulation.py` (target/source matching, influence metrics), `plot_jg.py` (plots) and `behavior.py`/`behavior_photostim.py` (behavior-only sessions) all work on `AnnData` objects or VR DataFrames. Functions that take `obs_key` / `var_key` filter rows with a dict of column→value (`functions.fetch_index`).

**Package import.** `__init__.py` imports the submodules under short aliases (`sess`, `an`, `pl`, `ps`, `bp`), and some modules rely on `from mouse_imaging import *` or `from mouse_imaging import sess`. Several modules call `importlib.reload(...)` on their dependencies at import time to support notebook workflows.

## Migration state (O2 → spinoza)

- `define_path` currently defines only raw, sync, virmen, metadata/session/adata and `suite2p_dir` paths. The `Session` loaders still read path keys that it no longer defines: `F_npy`, `Fneu_npy`, `stat_npy`, `Fc_oasis_mat`, `meanRef`, `cellpose` and `var_pickle`. Those loaders also loop over planes **1-based** (`range(1, nslices+1)`), whereas the new suite2p output is 0-based `plane0..N`. Building a `Session` will fail until these keys are added back and plane indexing is reconciled.
- O2-specific code is still present: `sbatch_*` helpers point at `~/code/preprocess_2p/*.slurm` (not in this repo), `update_oasis_file` uses an `/n/data2/...` path and references an undefined `oasis` module, and `plot_animation.slurm` uses `/n/app/conda2`.
- `options.default_ops()` uses `np` from an `import numpy as np` further down the module. This works only because the function is called after the module has finished loading.
