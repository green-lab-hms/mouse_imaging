# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`mouse_imaging` is a Python package for analyzing two-photon calcium imaging of mice running virtual-reality (ViRMEn) tasks, plus optional holographic photostimulation. It turns raw ScanImage TIFFs, ViRMEn behavior files and DAQ sync files into one `AnnData` object per session, which the analysis and plotting modules then use.

The code was originally written for the HMS O2 cluster (`/n/data2/...`, `/n/scratch/...`) and is being migrated to the `spinoza` cluster. Expect leftover O2 paths and code that depends on files the new layout doesn't produce yet (see "Migration state" below).

## Environment and running

- Conda env: `mouse_imaging`, at `~/.conda/envs/mouse_imaging`, with suite2p 1.1 and torch. The package is installed in editable mode (`pip install -e .`), so code changes take effect without reinstalling.
- Install: `conda env create -f environment.yml` creates the env and runs `pip install -e .[suite2p,notebook]`. Into an existing env, run `pip install -e .` (core) or `pip install -e ".[suite2p]"` (preprocessing). Dependencies are in `pyproject.toml`.
- The repo root *is* the package: `pyproject.toml` maps it with `package-dir = {"mouse_imaging" = "."}` instead of using a `src/` layout. A regular (non-editable) `pip install .` or `pip wheel .` fails when run inside the repo on the network share (`Directory not empty` when cleaning up `build/`); build from a copy on local disk if you need a wheel.
- Activate on spinoza: `source /molbio/hpc/apps/miniforge3/etc/profile.d/conda.sh && conda activate mouse_imaging`
- Cluster: SLURM, a single node `spinoza` (partition `defq`, 96 CPUs, about 250 GB RAM, **no GPU**, so suite2p/cellpose run with `torch_device='cpu'`).
- There are no tests, linter or build step. To check your changes, import the module and run the relevant script:
  ```bash
  cd ~/code && python -c "import mouse_imaging.preprocess"
  ```

### Entry points

| Command | What it does |
|---|---|
| `sbatch preprocess.slurm <mouse> <date> [session] [steps]` | Runs `preprocess.py`: step `suite2p` (all planes together), then step `anndata` (builds the `Session` and AnnData and saves `adata.h5ad`). `steps` is comma-separated and defaults to `suite2p,anndata`; pass `anndata` to rebuild the AnnData without rerunning suite2p. The log goes to `preprocess_<jobid>.log` in the directory you submit from. |
| `python -m mouse_imaging.preprocess --mouse M --date D --steps anndata` | The same as above for step 2 only, run directly. It takes a few minutes, so it doesn't need SLURM. The SLURM script uses this module form too, so it works without a clone. |
| `python session.py --mouse M --date D --update` | Re-runs `update_function` (photostim influence, tuning, regressions) on an existing `adata.h5ad`. |
| `sbatch plot_animation.slurm ...` | Renders VR and activity movies (`plot_animation.py`). This file still uses the O2 conda path. |

`README.md` is the user-facing guide: install routes, a pipeline diagram and the AnnData layout. `examples/example_session.ipynb` is an executed walkthrough on JG6/260929/session_1. Re-execute it after changing APIs it uses: `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=mouse_imaging examples/example_session.ipynb`.

## Data layout

Roots are set in `options.default_ops()`; `session.define_path()` builds every file path from them:

- Raw: `/data/green_lab/shared/data/raw/{twophoton,virmen,sync}/<mouse>/<date>/<session>/`
- Derived: `/data/green_lab/shared/data/derived/twophoton/<mouse>/<date>/<session>/`, containing `suite2p/plane0..N/`, `adata.h5ad` and `session.pickle`
- Sync files are WinEDR `.EDR` files (`functions.import_edr`), with channels `ScanImageTrigger` (frame clock), `Virmen` (one pulse per ViRMEn iteration), `Lick detection` and unconnected `Ground` channels. Older sessions used `.abf`/`.h5`.

Sessions are identified by a key dict, `{'mouse', 'date', 'session'}`. Dates are `YYMMDD` strings and sessions are `session_N`. Raw TIFF filenames encode acquisition info, for example `920nm_G1R1_V1_L123_00001_00001.tif` gives wavelength, emission filters (color+number pairs), region and depth. `parse_si_filename` parses these names.

## Architecture

**Configuration ("ops").** `options.py` defines `default_ops()`, a plain dict holding paths, behavior constants, sync column mapping, ViRMEn column names and `ops['suite2p_settings']`. That last one holds **only nested overrides** to suite2p 1.x `default_settings()`; `preprocess.suite2p_settings()` merges them and picks the torch device. Variants are selected by function name: scripts take `--ops <name>` and call `getattr(options, name)()`. `options.py` also holds cell-type calling and post-processing functions (`call_celltypes_*`, `process_*`), and `session.main` finds these through `ops['process_fcn']`. `default_ops` imports nothing, so analysis-only environments don't need suite2p or torch. Even `from suite2p.parameters import ...` runs suite2p's `__init__`, which loads the whole package and torch, about 8 s. Step 2 reads the baseline settings suite2p actually used from `suite2p/settings.npy`.

**Metadata.** `session.get_metadata(path)` reads the ScanImage header of the first raw TIFF (`parse_si_metadata` flattens it into `'SI.hX.y'` keys) and exposes `nslices`, `nflyback`, `nchannels`, `Ly`, `Lx` and `volume_rate`. Multi-element SI values such as `[1;2]` stay strings. Use these derived keys rather than raw SI keys for slice counts: in ScanImage 2026, `SI.hStackManager.numSlices` is 1 for fast-z stacks, `actualNumSlices` holds the real count, and `SI.hFastZ.numFramesPerVolume` no longer exists.

**Preprocessing (`preprocess.py`).** This passes all planes to suite2p in a single `run_s2p(settings=..., db=...)` call. Flyback frames count toward `nplanes` and are skipped with `ignore_flyback`. suite2p's plane folders are **0-based**.

**Session assembly (`session.Session`).** The constructor runs the whole pipeline:
1. `Sync` loads the DAQ file (`.EDR`, `.abf` or `.h5`), renames columns to canonical names (`Virmen`, `ScanImage`, `Ball_*`, `Licks`) and finds VR-frame and imaging-volume indices from the clock edges. Flyback frames are removed using `nslices`/`nflyback`.
2. Loads the ViRMEn `sessionData.mat` (`load_vr`) and aligns it to sync. `maze_id` reads the maze name from `experName` in that file.
3. Loads suite2p `stat` and activity for planes `0..nslices-1`. Columns are named `plane{p}_source{i}` (0-based); `X` and the `dcnv` layer hold suite2p `spks.npy`, and the `dF` layer holds dF/F from `F.npy`, using the same baseline settings as suite2p (`suite2p_settings['dcnv_preprocess']`). Then it crops imaging and VR to their overlap.
4. Builds `obs` (per imaging volume: VR variables resampled to the volume times) and `var`. Per source, `var` has `G`/`R` intensities (ROI minus surrounding shell, measured on suite2p's `meanImg`/`meanImg_chan2`), suite2p `iscell`/`redcell` with their probabilities, `plane`, and position.
5. Optionally runs photostim processing (`photostimulation.process_photostim_session`).
6. Runs the maze-specific variables in `importlib.import_module('mouse_imaging.' + ops['env']).main(self, ...)`. `env` is a module name such as `tmaze` or `wideLinearTrack`.
7. Pickles itself. `load_as_anndata` wraps the result as `AnnData(X, obs, var, uns, layers)`, and `uns` carries `path`, `metadata`, `ops` and `vr`. `uns` goes through `functions.to_h5ad_safe` first, because h5ad can't store `Path`s, `None` or lists of dicts.

`session.main` then applies `select_cells` (`isnotclipped`, `isnotnearedge`, `iscell`) and saves `adata.h5ad`.

**Downstream.** `analysis.py` (binning, triggered averages `trigger_*`, tuning, regression), `photostimulation.py` (target/source matching, influence metrics), `plot_jg.py` (plots) and `behavior.py`/`behavior_photostim.py` (behavior-only sessions) all work on `AnnData` objects or VR DataFrames. Functions that take `obs_key` / `var_key` filter rows with a dict of column→value (`functions.fetch_index`).

**Package import.** Keep `import mouse_imaging` fast. Slow libraries such as `umap` (about 7 s), `suite2p` and `torch` are imported inside the functions that need them. `__init__.py` imports the submodules under short aliases (`sess`, `an`, `pl`, `ps`, `bp`), and some modules rely on `from mouse_imaging import *` or `from mouse_imaging import sess`. Several modules call `importlib.reload(...)` on their dependencies at import time to support notebook workflows.

## Migration state (O2 → spinoza)

- `preprocess.py` (suite2p plus session assembly) works on spinoza data. It was tested on JG6/260929/session_1.
- Defaults to `env=None`: no maze-specific `obs` variables are computed yet. `tmaze.main` assumes cued T-maze worlds, and `wideLinearTrack.py` has no `main()`, so linear-track sessions need a new env module.
- These still use the old O2 layout or data and will fail if called:
  - `update_function` / `session.py --update`: photostim influence, plus regressions on `h_mt_error` and `pitch`
  - photostimulation processing (`fov_microns`, `slm_photostim`, `df.append`)
  - `update_session` / `update_oasis_file`
  - the `sbatch_*` helpers (they point at `~/code/preprocess_2p`)
  - `plot_jg.py` photostim plots (`plot_target_locations*`), which still read `meanRef`. `specificity_vs_cutoff` no longer filters on `B_spatial_corr`; that came from cellpose and is commented out with a TODO.
  - `plot_animation.slurm` (O2 conda path)
- With `detection.chan2_threshold=0.25`, suite2p's `redcell` flag was True for nearly all ROIs on JG6/260929, where `redcell_prob` ranged about 0.44–0.79. Use `redcell_prob` or `R`, or retune the threshold.
- `options.default_ops()` uses `np` from an `import numpy as np` further down the module. This works only because the function is called after the module has finished loading.
