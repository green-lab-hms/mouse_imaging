# mouse_imaging

Preprocessing and analysis of two-photon calcium imaging in mice running virtual-reality (ViRMEn) tasks.

`mouse_imaging` turns one recording session into a single [AnnData](https://anndata.readthedocs.io) file. A session's raw data is ScanImage TIFFs, a ViRMEn behavior file and a DAQ sync file. The AnnData file holds neural activity, behavior aligned to each imaging volume, and per-cell properties, so a session can be loaded and analyzed with one line of code.

## Installation

There are two ways to install, depending on whether you'll change the code.

### Option 1: clone the repository (to edit the code)

```bash
git clone https://github.com/green-lab-hms/mouse_imaging.git
cd mouse_imaging
conda env create -f environment.yml
conda activate mouse_imaging
```

`environment.yml` creates a conda env called `mouse_imaging` with Python 3.11. It then runs `pip install -e ".[suite2p,notebook]"`, which installs the package in editable mode, so your changes to the code take effect without reinstalling. To install into an environment you already have, run one of these from the repo directory:

```bash
pip install -e .              # analysis only
pip install -e ".[suite2p]"   # plus suite2p and torch, needed to preprocess raw data
```

### Option 2: install from GitHub (to use the package as is)

```bash
conda create -n mouse_imaging python=3.11
conda activate mouse_imaging
pip install "mouse_imaging[suite2p,notebook] @ git+https://github.com/green-lab-hms/mouse_imaging.git"
```

To install a specific branch or commit, add `@<branch-or-commit>` after `.git`.

### Optional extras

Both routes above install the two optional extras, `suite2p` and `notebook`. They're included by `environment.yml` in option 1 and by `[suite2p,notebook]` in option 2. To install only the analysis code, leave the extras out: `pip install -e .` or `pip install "mouse_imaging @ git+..."`.

| Extra | Adds | Needed for |
|---|---|---|
| `suite2p` | suite2p ≥ 1.1 and torch (several GB) | Step 1 of preprocessing: motion correction, cell detection, deconvolution |
| `notebook` | ipykernel | Running the example notebook |

To check that the extras are installed:

```bash
python -c "import mouse_imaging, suite2p, ipykernel; print('ok')"
```

To use the environment in Jupyter or VS Code notebooks, register it as a kernel once:

```bash
python -m ipykernel install --user --name mouse_imaging --display-name "Python (mouse_imaging)"
```

## Configuration

The package needs to know where raw data is and where to write preprocessed data. The built-in defaults are the green lab's paths on spinoza, so lab members on spinoza don't need to configure anything.

On another system, create a config file with your paths:

```bash
mkdir -p ~/.config/mouse_imaging
cp config.example.toml ~/.config/mouse_imaging/config.toml   # from the repo; or create the file by hand
```

```toml
# ~/.config/mouse_imaging/config.toml
[paths]
raw_root = "/path/to/raw"           # contains twophoton/, virmen/ and sync/
derived_root = "/path/to/derived"   # preprocessing output: <mouse>/<date>/<session>/
```

Settings are read from the first of these that exists:
1. the file named by the `MOUSE_IMAGING_CONFIG` environment variable, e.g. a shared lab config: `export MOUSE_IMAGING_CONFIG=/shared/mouse_imaging.toml`
2. `~/.config/mouse_imaging/config.toml`
3. the built-in defaults

A config file only needs the keys it changes; the rest keep their defaults. If a configured folder doesn't exist, loading or preprocessing stops with an error naming the setting and the config file it came from. To see which settings are in use:

```bash
python -c "from mouse_imaging import config; print(config.describe_source()); print(config.load_config())"
```

On a new cluster, also check the `#SBATCH` lines at the top of `preprocess.slurm`: partition, CPUs, memory and time limit. You can edit them, or override them when submitting, e.g. `sbatch -p <partition> preprocess.slurm ...`. The script uses the `conda` on your `PATH`. If batch jobs can't find it, pass your conda location: `sbatch --export=ALL,CONDA_BASE=/path/to/miniforge3 preprocess.slurm ...`.

## How it works

### Pipeline

```
 Raw data (<raw_root>)                                     Derived data (<derived_root>/<mouse>/<date>/<session>/)
 ─────────────────────                                     ──────────────────────────────────────────────────────

 twophoton/<mouse>/<date>/<session>/*.tif ──┐
   ScanImage TIFFs, all planes and channels │   Step 1: suite2p
                                            ├────────────────────► suite2p/plane0 … planeN/
                                            │   registration,        F.npy, spks.npy, stat.npy,
                                            │   cell detection,      iscell.npy, redcell.npy, ops.npy
                                            │   deconvolution                     │
                                            │                                     │
 sync/<mouse>/<date>/session_00N.EDR ───────┤   Step 2: anndata                   │
   frame clock, ViRMEn clock, licks         ├────────────────────► adata.h5ad ◄───┘
                                            │   align imaging and    (one AnnData per session)
 virmen/<mouse>/<date>/<session>/           │   behavior on the
   sessionData.mat (position, speed, ...) ──┘   sync clock
```

1. **Metadata:** `session.get_metadata` reads the ScanImage header of the first TIFF: number of planes and flyback frames, channels, frame size and volume rate. Acquisition details also come from the filename: for `920nm_G1R1_V1_L123_00001_00001.tif`, that's the laser wavelength, emission filters (G1, R1) and region (V1).
2. **Step 1, suite2p:** `preprocess.py` runs suite2p on all planes together and skips the flyback planes. Settings are in `options.default_ops()['suite2p_settings']` as overrides of suite2p's defaults.
3. **Step 2, anndata:** `session.Session` combines the outputs.
   - It reads the sync file and finds when each imaging volume and each ViRMEn iteration happened.
   - It resamples behavior to the imaging volume times, and trims both to the period when behavior was running.
   - For each cell, it measures green and red brightness on the mean images.
   - `session.main` keeps cells that suite2p classifies as cells, that aren't near the image edge and that aren't saturated, then saves `adata.h5ad`.

### The AnnData object

Each session is one AnnData object. Rows are **imaging volumes** (time points) and columns are **cells**:

```
                                 adata.var  (one row per cell)
                                 plane, x, y, G, R, iscell, iscell_prob, redcell, redcell_prob, ...
                                 ┌────────────────┬────────────────┬─────┬────────────────┐
                                 │ plane0_source0 │ plane0_source1 │ ... │ plane2_source9 │
              ┌──────────────────┼────────────────┴────────────────┴─────┴────────────────┤
 adata.obs    │ t=48.17 y=25.0   │                                                         │
 (one row per │ t=48.30 y=34.3   │   adata.X              deconvolved activity (spks)      │
 imaging      │ t=48.44 y=41.7   │   adata.layers['dcnv'] same as X                        │
 volume):     │   ...            │   adata.layers['dF']   dF/F                             │
 time and     │ t, dt, y, dy,    │                                                         │
 behavior     │ trial, lick, ... │   shape: n_volumes × n_cells                            │
 (ViRMEn)     └──────────────────┴─────────────────────────────────────────────────────────┘

 adata.uns   metadata  ScanImage header + mouse, date, session, region, maze, nslices, volume_rate, ...
             ops       options used to build the session (paths, suite2p setting overrides, ...)
             path      file paths for this session (raw data, suite2p output, h5ad files)
             vr        full-rate ViRMEn table (about 60 Hz), on the sync clock
             triggers  sync event times
```

- **Cell names:** `plane{p}_source{i}`, where `p` is the 0-based suite2p plane and `i` is the ROI's index in that plane's `stat.npy`. `sess.fetch_cell_stat(adata, name)` returns a cell's suite2p footprint.
- **Behavior:** `obs` holds the behavioral data. Each ViRMEn variable (position `x`/`y`, heading `h`, velocity `dx`/`dy`/`dh`, `trial`, `inITI`, `reward`, `lick`, plus any extra `user0..N` channels) is resampled to the time of each imaging volume. The full-rate ViRMEn table is in `uns['vr']`.
- **Time:** `obs['t']` is in seconds on the sync clock. Because volumes are ~7.5 Hz with 3 planes, each row is one volume, not one frame.
- **Red and green brightness:** `var['G']` and `var['R']` are the mean brightness inside the cell minus the mean in a 2-pixel ring around it, measured on suite2p's registered mean image of each channel.
- **Derived layers:** `options.preprocess_activity(adata)` adds smoothed and normalized layers (`dcnv_0.25sigma`, `dcnv_norm`, `dcnv_0.25sigma_norm`). Many analysis functions expect these.

## Usage

### Preprocess a session

A **session** is one recording: one continuous ScanImage acquisition with its matching behavior and sync files. A mouse can have several sessions on the same day, numbered `session_1`, `session_2`, ... Each session is identified by three arguments, which name the raw data folders:

| Argument | Example | Meaning |
|---|---|---|
| `mouse` | `JG6` | Mouse ID |
| `date` | `260929` | Recording date, `YYMMDD` |
| `session` | `session_1` | Recording folder for that day; defaults to `session_1` |

For example, `JG6 260929 session_2` reads:
- `<raw_root>/twophoton/JG6/260929/session_2/*.tif`
- `<raw_root>/virmen/JG6/260929/session_2/sessionData.mat`
- `<raw_root>/sync/JG6/260929/session_002.EDR` (same number, zero-padded)

It writes to `<derived_root>/JG6/260929/session_2/`. Raw data has to be in this folder structure before preprocessing.

Preprocessing has two steps, which can be run together or separately:

| Step | Does | Output |
|---|---|---|
| `suite2p` | Motion correction, cell detection, fluorescence extraction and deconvolution. Slow: about 20 minutes on CPU for the 22-minute, 3-plane test session (JG6/260929). | `suite2p/plane0..N/` |
| `anndata` | Aligns suite2p output with behavior and sync, measures cell properties and saves the AnnData. About 1 minute. | `adata.h5ad` |

On spinoza, from the repo directory, or anywhere if you copy `preprocess.slurm`:

```bash
sbatch preprocess.slurm <mouse> <date> [session] [steps]
# e.g.
sbatch preprocess.slurm JG6 260929                     # both steps, session_1
sbatch preprocess.slurm JG6 260929 session_2           # both steps, the day's second session
sbatch preprocess.slurm JG6 260929 session_1 anndata   # rebuild adata.h5ad only, from existing suite2p output
```

The script runs `python -m mouse_imaging.preprocess`, which you can also call directly:

```bash
python -m mouse_imaging.preprocess --mouse JG6 --date 260929 --session session_1 --steps anndata
```

### Load and analyze

```python
from mouse_imaging import sess, an, pl, options

# Load one or more sessions of a mouse as a list of AnnData objects
adatas = sess.load_imaging_sessions('JG6', dates=['260929',])
adatas = [options.preprocess_activity(adata) for adata in adatas]

# Mean activity of running periods in 20-unit position bins, per cell
tunings = [an.binX1(adata, obs='y', bins=range(0, 401, 20), obs_key={'dy': '>5'}) for adata in adatas]
```

Each date loads `session_1` by default. For another session that day, give a dict instead of a string: `dates=['260929', {'date': '260929', 'session': 'session_2'}]`. Sessions without a saved `adata.h5ad` are skipped with a warning. Many `pl.*` plotting functions take the list directly. To load a single session, use `sess.load_as_anndata('JG6', '260929', 'session_1')`.

Most functions select data with `obs_key` (which volumes) and `var_key` (which cells). These are dicts of column → value; a string starting with `<`, `>`, `=` or `!` is a comparison, e.g. `var_key={'plane': 0, 'iscell_prob': '>0.8'}`.

See [examples/example_session.ipynb](examples/example_session.ipynb) for a full walkthrough:
- what's in the AnnData object
- the field of view and cell footprints
- behavior
- activity along the track, with cross-validation
- activity aligned to lap starts
- running modulation
- the red channel
- individual cells

### Package layout

| Module | Alias | Contents |
|---|---|---|
| `preprocess.py` | | Command-line pipeline: step 1 `suite2p`, step 2 `anndata` |
| `session.py` | `sess` | Paths, metadata, sync, `Session` assembly, loading AnnData |
| `options.py` | | `default_ops()` settings, activity preprocessing, cell-type calling |
| `analysis.py` | `an` | Binning, tuning, event-triggered activity, regression |
| `plot_jg.py` | `pl` | Plotting: fields of view, cells, binned and triggered activity |
| `config.py` | | System settings: data locations, read from the config file |
| `functions.py` | `fc` | Low-level helpers: file readers (EDR, ABF, H5), indexing, filtering |
| `tmaze.py`, `wideLinearTrack.py`, `behavior.py` | | Maze-specific behavior variables |
| `photostimulation.py`, `behavior_photostim.py` | `ps`, `bp` | Holographic photostimulation analysis |

## Status

The code is being ported from the HMS O2 cluster to spinoza. Step 1 and step 2 of preprocessing, loading and the analyses in the example notebook work on the new data. These are not ported yet:
- maze-specific behavior variables (the `env` step; `default_ops` currently sets `env=None`)
- the `session.py --update` analyses
- photostimulation processing

See the "Migration state" section of [CLAUDE.md](CLAUDE.md) for details.
