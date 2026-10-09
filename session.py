import glob, os, copy, json, warnings, subprocess, time, re, datetime

import pandas as pd
import numpy as np
import anndata
import scipy.io
import scipy.stats
import scipy.ndimage
from tifffile import imread
from ScanImageTiffReader import ScanImageTiffReader

from mouse_imaging import functions as fc
from mouse_imaging import options
from mouse_imaging import config


import matplotlib.pyplot as plt
import matplotlib as mpl

import importlib
importlib.reload(options)

regex_1filter = "^[BGR]\d$"
regex_2filters = "^[BGR]\d[BGR]\d$"
chan_dict = {'B': 0, 'G': 1, 'R': 2}
chan_dict_r = {val: key for key, val in chan_dict.items()}

# Session parameters
def parse_si_metadata(metadata_str):
    part1, part2 = metadata_str.split('\n\n')
    metadata = {}
    for line in part1.split('\n'):
        prop, val = line.split(' = ')
        if val[0] == "'":
            val = val[1:-1]
        elif val in ['true', 'false']:
            val = (val == 'true')
        elif val[0].isdigit() or val == 'NaN':
            val = float(val)
        metadata[prop] = val
        
    metadata['RoiGroups'] = json.loads(part2)['RoiGroups']

    return metadata

def parse_si_filename(si_tif, functional_filter='G'):
    metadata = {}
    # Parse metadata within filename
    
    if si_tif[-10:] == '_mean.tiff':
        entries = os.path.basename(si_tif).split('_')[:-2]
        metadata['file_basename'] = '_'.join(os.path.basename(si_tif).split('_')[:-3])
    else:
        entries = os.path.basename(si_tif).split('_')[:-2]
        metadata['file_basename'] = '_'.join(os.path.basename(si_tif).split('_')[:-2])
    for entry in entries:
        if entry.endswith('nm'):
            metadata['laserWavelength_nm'] = float(entry.rstrip('nm'))
        elif entry.endswith('um'):
            depth = entry.rstrip('um')
            metadata['depth_um'] = float(depth) if depth.isdigit() else np.nan
        elif re.match(regex_1filter, entry):
            metadata['filter_id'] = entry
            metadata['em_filters'] = [entry]
        elif re.match(regex_2filters, entry):
            metadata['filter_id'] = entry
            metadata['em_filters'] = [entry[:2], entry[2:4]]
        elif entry != 'session' and re.match('\D', entry) and 'region' not in metadata:
            metadata['region'] = entry # first non-numeric entry, e.g. V1 in 920nm_G1R1_V1_L123

    metadata['channels'] = [em_filter[0] for em_filter in metadata['em_filters']]
    if functional_filter in metadata['channels']:
        metadata['functional_chan'] = metadata['channels'].index(functional_filter) + 1

    return metadata

def get_metadata(path, functional_filter='G', recompute=False, update=True):
    assert (path['raw_image1_tif'] is not None) and path['raw_image1_tif'].is_file()

    if os.path.isfile(path['metadata_pickle']) and ( (not recompute) or update):
        metadata = fc.load_pickle(path['metadata_pickle'])
        if update:
            metadata_new = get_metadata(path, functional_filter=functional_filter, recompute=True, update=False)
            metadata.update(metadata_new)

    else:
        def parse_si_path(si_tif):
            metadata = {}
            mouse, date, session = si_tif.split('/')[-4:-1] # .../twophoton/<mouse>/<date>/<session>/<file>.tif
            metadata['mouse'] = mouse
            metadata['date'] = date
            metadata['session'] = session
            return metadata

        def available_channels(metadata):
            filterxs = [key for key in metadata.keys() if re.match('filter\d', key)]
            channels = []
            for filterx in filterxs:
                channels.extend( metadata[filterx]['channels'] )
            channels = sorted(list(set(channels)))
            return channels

        def available_filter_ids(metadata):
            filterxs = [key for key in metadata.keys() if re.match('filter\d', key)]
            filters = [metadata[filterx]['filter_id'] for filterx in filterxs]
            return filters

        # def filterx_filename(path, filterx):
        #     filter_keys_raw = [key for key in path['meanRef'].keys() if not key[-8:]=='demixed']
        #     for filter_key in filter_keys_raw:
        #         files = path['meanRef'][filter_key]
        #         if filterx in files[0]:
        #             return files[0]

        def n_saved_channels(metadata):
            val = metadata['SI.hChannels.channelSave']
            if isinstance(val, float):          # single channel, e.g. '1' -> 1.0
                return 1
            return len(val.strip('[]').replace(';', ' ').split())   # e.g. '[1;2]'

        
        
        metadata = {}
        
        filter1_tif = str(path['raw_image1_tif'])
        # Parse ScanImage metadata within tif file
        img = ScanImageTiffReader(str(filter1_tif))
        # metadata['scanimage_str'] = img.metadata()
        metadata.update(parse_si_metadata(img.metadata()))
        metadata.update(parse_si_path(filter1_tif))
        metadata['is_photostim'] = metadata['RoiGroups']['photostimRoiGroups'] is not None

        # Make some useful parameters more easily accessible
        metadata['nslices'] = int(metadata['SI.hStackManager.actualNumSlices'])
        metadata['nflyback'] = int(metadata['SI.hFastZ.numDiscardFlybackFrames'])
        metadata['nchannels'] = n_saved_channels(metadata)
        metadata['Ly'] = int(metadata['SI.hRoiManager.linesPerFrame'])
        metadata['Lx'] = int(metadata['SI.hRoiManager.pixelsPerLine'])
        metadata['volume_rate'] = float(metadata['SI.hRoiManager.scanVolumeRate'])
        metadata['dt'] = 1. / metadata['volume_rate']
        
        metadata['filter1'] = parse_si_filename(filter1_tif, functional_filter)
        metadata['region'] = metadata['filter1'].get('region')
        
        # Parse metadata from filter2, if present
        # filter2_tif = filterx_filename(path, 'filter2')
        # if filter2_tif:
        #     metadata['filter2'] = parse_si_filename(filter2_tif, functional_filter)

        # Get channels that are available
        metadata['available_channels'] = available_channels(metadata)
        metadata['available_filters'] = available_filter_ids(metadata)

        metadata['updates'] = ['suite2p_script_glob_sorted_2022-10-11']

    return metadata

def mousedate_tokey(mouse, date):
    if type(date) is dict:
        key = dict(mouse=mouse, **date)
    else:
        key = dict(mouse=mouse, date=date, session='session_1')
    return key

def mousedates_to_keys(mouse, dates, filter_key=None):
    keys = [mousedate_tokey(mouse=mouse, date=date) for date in dates]
    if filter_key is not None:
        keys = filter_keys(keys, filter_key)
    return keys

def adata_to_key(adata):
    key = {key: adata.uns['metadata'][key] for key in ['mouse', 'date', 'session']}
    return key

def update_metadata(adata):
    try:
        key = metadata_to_key(adata.uns['metadata'])
        path = define_path(**key, recompute=False, update=True)
    except:
        path = adata.uns['path']
    adata.uns['metadata'] = get_metadata(path)

def define_path(mouse=None, date=None, session='session_1', ops=None, makedir=False, recompute=False, update=True):
    if ops is None:
        ops = options.default_ops()
    config.check_dir(ops['raw_root'], 'raw_root')
    config.check_dir(ops['preprocessed_root'], 'derived_root')
    path = {}
    # Preprocessed output
    path['raw_root'] = ops['raw_root']
    path['preprocessed_root'] = ops['preprocessed_root']
    path['preprocessed_dir'] = path['preprocessed_root'] / mouse / date / session

    # Virmen path
    path['virmen_dir'] = path['raw_root'] / 'virmen' / mouse / date / session
    path['virmen_mat'] = path['virmen_dir'] / 'sessionData.mat'
    path['maze_id_txt'] = path['virmen_dir'] / 'maze_id.txt'
    path['frameGrabs_dir'] = path['virmen_dir'] / 'frameGrabs'
    path['frameGrabs_mat'] = path['frameGrabs_dir'] / f'Trial{{trial}}.mat'

    # Two-photon and sync paths
    if ops['imaging']:
    # Two-photon path
        path['twophoton_dir'] = path['raw_root'] / 'twophoton' / mouse / date / session
        raw_tifs = sorted(list(path['twophoton_dir'].glob('*.tif')))
        path['raw_image1_tif'] = raw_tifs[0] if raw_tifs else None
            
        # Sync path
        path['sync_dir'] = path['raw_root'] / 'sync' / mouse / date
        sync_files = list(path['sync_dir'].glob(f"session_{int(session.split('_')[-1]):03d}.*"))
        path['sync'] = sync_files[0] if sync_files else None
        
        path['metadata_pickle'] = path['preprocessed_dir'] / 'metadata.pickle'
        path['session_pickle'] = path['preprocessed_dir'] / 'session.pickle'
        path['adata_h5ad'] = path['preprocessed_dir'] / f'adata.h5ad'
        path['adata_h5ad_backup'] = path['preprocessed_dir'] / f'adata.h5ad.backup'
        path['adata_allsources_h5ad'] = path['preprocessed_dir'] / f'adata_allsources.h5ad'
        
        path['var_pickle'] = path['preprocessed_dir'] / 'var.pickle'

        # Suite2p output, one folder per plane (0-based). Format with plane=
        path['suite2p_dir'] = path['preprocessed_dir'] / 'suite2p'
        path['suite2p_settings_npy'] = path['suite2p_dir'] / 'settings.npy' # settings suite2p ran with
        plane_dir = str(path['suite2p_dir'] / 'plane{plane}')
        for filekey, filename in [('F_npy', 'F.npy'), ('Fneu_npy', 'Fneu.npy'), ('spks_npy', 'spks.npy'), ('stat_npy', 'stat.npy'),
                                  ('iscell_npy', 'iscell.npy'), ('redcell_npy', 'redcell.npy'), ('ops_npy', 'ops.npy')]:
            path[filekey] = os.path.join(plane_dir, filename)

    if makedir:
        os.makedirs(path['preprocessed_dir'], exist_ok=True)

    return path

def update_path(adata):
    key = adata_to_key(adata)
    adata.uns['path'] = define_path(**key, recompute=False, update=True)

def maze_id(path, strip=True, load_if_not_saved=True, force_load=False):
    if os.path.isfile(path['maze_id_txt']) and (not force_load):
        with open(path['maze_id_txt'], 'r') as fh:
            maze = fh.read()
    elif load_if_not_saved:
        # Maze name is saved as experName in sessionData.mat
        try:
            mat = scipy.io.loadmat(path['virmen_mat'], variable_names=['experName'])
            maze = str(np.squeeze(mat['experName'])).strip()
        except (FileNotFoundError, KeyError, ValueError):
            warnings.warn('Could not read maze name (experName) from virmen file.')
            maze = ''

        os.makedirs(os.path.dirname(path['maze_id_txt']), exist_ok=True)
        with open(path['maze_id_txt'], 'w') as fh:
            fh.write(maze)
    else:
        maze = ''
    if strip:
        maze = maze.split('dw_')[-1].split('_updated')[0]
    return maze

# Match cells across days
def match_var_names(adata, sourcemap):
    key = adata_to_key(adata)
    date = adata.uns['metadata']['date']
    ref_to_adata = sourcemap[str(key)].dropna()
    adata_to_ref = pd.Series(ref_to_adata.index.values, index=ref_to_adata)
    var_names = adata.var_names.map(adata_to_ref)
    return var_names

# Check if session files are updated
def update_file(key, filename, recompute=False, min_datetime=None, print_filename=True, update_fcn=None):
    isfile = os.path.isfile(filename)
    if isfile:
        if (not recompute) and (min_datetime is not None):
            timestamp = os.path.getmtime(filename)
            modified_datetime = datetime.datetime.fromtimestamp(timestamp)
            if modified_datetime < min_datetime:
                recompute = True
    else:
        recompute = True
    
    if recompute:
        if print_filename:
            print(filename)
            # print('{mouse} {date} {session}'.format(**key))

        if update_fcn is not None:
            update_fcn(key)

def update_session(key, filekey, ops, **kwargs):
    path = define_path(**key, ops=ops)
    if filekey == 'Fc_oasis_mat':
        if os.path.isfile(path['metadata_pickle']):
            md = fc.load_pickle(path['metadata_pickle'])
        else:
            md = get_metadata(path)
        planes = list(range(1, md['nslices']+1))
        planes = [1]
        for plane in planes:
            filename = path[filekey].format(plane=plane)
            update_file(key, filename, **kwargs)
    else:
        filename = path[filekey]
        update_file(key, filename, **kwargs)

def date_to_datedict(date):
    if type(date) is str:
        return dict(date=date, session='session_1')
    elif type(date) is dict:
        return date
    else:
        raise TypeError
    
def update_sessions(mouse, dates, **kwargs):
    keys = [dict(mouse=mouse, **date_to_datedict(date)) for date in dates]
    for key in keys:
        update_session(key, **kwargs)

def update_oasis_file(key, filename, ops):
    filename2 = filename.split('/n/data2/hms/neurobio/harvey/jonathan/data/imaging/')[1]
    mouse, date, session, plane_, _, oasis_file = filename2.split(os.path.sep)
    # key = dict(mouse=mouse, date=date, session=session)
    plane = int(plane_.split('plane')[1])
    print(plane)
    oasis.oasis_plane(key, ops=ops, plane=plane)

def sbatch_load_anndata(key, ops_name):
    shell_script = """
    sbatch ~/code/preprocess_2p/load_anndata.slurm {mouse} {date} {session} {ops_name}
    """.format(**key, ops_name=ops_name)
    print(shell_script)
    out = subprocess.call(shell_script, shell=True)

def sbatch_load_anndatas(mouse, dates, ops_name):
    keys = mousedates_to_keys(mouse, dates)
    for key in keys:
        sbatch_load_anndata(key, ops_name)

def sbatch_register_sessions(mouse, dates, min_corr=0.4):
    dates_str = ' '.join(dates)
    shell_script = f'sbatch ~/code/preprocess_2p/register_sessions.slurm {mouse} "{dates_str}" {min_corr}'
    print(shell_script)
    out = subprocess.call(shell_script, shell=True)

def sbatch_update_anndata(key,):
    shell_script = """
    sbatch ~/code/preprocess_2p/update_anndata.slurm {mouse} {date} {session}
    """.format(**key)
    print(shell_script)
    out = subprocess.call(shell_script, shell=True)

def sbatch_update_anndatas(mouse, dates):
    keys = mousedates_to_keys(mouse=mouse, dates=dates)
    for key in keys:
        sbatch_update_anndata(key,)

def save_adata(adata, adata_filekey='adata_h5ad'):
    adata.write(adata.uns['path'][adata_filekey])

# Preprocessing
def F_baseline(F, ops):
    """ 
    Copied from suite2p source code, except returns baseline instead of corrected F.
    
    Parameters
    ----------------

    F : float, 2D array
        size [neurons x time], in pipeline uses neuropil-subtracted fluorescence

    baseline : str
        setting that describes how to compute the baseline of each trace

    win_baseline : float
        window (in seconds) for max filter

    sig_baseline : float
        width of Gaussian filter in seconds

    fs : float
        sampling rate per plane

    prctile_baseline : float
        percentile of trace to use as baseline if using `constant_prctile` for baseline
    
    Returns
    ----------------

    F : float, 2D array
        size [neurons x time], baseline-corrected fluorescence

    """
    
    if ops['baseline'] == 'maximin':
        win = int(ops['win_baseline']*ops['fs'])
        Flow = scipy.ndimage.gaussian_filter(F, [0., ops['sig_baseline']])
        Flow = scipy.ndimage.minimum_filter1d(Flow,    win)
        Flow = scipy.ndimage.maximum_filter1d(Flow,    win)
    elif ops['baseline'] == 'constant':
        Flow = scipy.ndimage.gaussian_filter(F, [0., ops['sig_baseline']])
        Flow = np.amin(Flow)
    elif ops['baseline'] == 'constant_prctile':
        Flow = np.percentile(F, ops['prctile_baseline'], axis=1)
        Flow = np.expand_dims(Flow, axis = 1)
    else:
        Flow = 0.

    return Flow

def preprocess_dF(path=None, ops=None, fs=None, plane=None):
    # Load raw traces
    F = np.load(path['F_npy'].format(plane=plane))

    # Subtract neuropil
    # Fneu = np.load(path['Fneu_npy'].format(plane=plane))
    # Fc = F - Fneu * ops['neucoeff']
    # JG 211101: Omit neuropil subtraction for dF/F because this ends up causing problems with negative baselines
    Fc = F

    # Compute dF/F, with the same baseline settings suite2p used before deconvolution
    suite2p_settings = np.load(path['suite2p_settings_npy'], allow_pickle=True).item()
    baseline_ops = {**suite2p_settings['dcnv_preprocess'], 'fs': fs}
    Fb = F_baseline(Fc, baseline_ops)
    Fb[Fb<=0] = np.nan
    dF = (Fc - Fb) / (Fb)
    return dF

# Loading and saving data
def load_hyperstack(tif_files):
    """
    Expecting each tif file to be from one slice, with
    each tif file having dimensions: (Ly, Lx, nchannels).
    """
    nslices = len(tif_files)
    Ly, Lx, nchannels = imread(tif_files[0]).shape
    hyperstack = np.zeros((nslices, nchannels, Ly, Lx))
    for islice, tif_file in enumerate(tif_files):
        img_slicei = imread(tif_file)
        for ichannel in range(nchannels):
            hyperstack[islice, ichannel] = img_slicei[:, :, ichannel]
    return hyperstack

def load_vr(session_mat, columns=None, playback=False):
    # Load data intoata dataframe
    mat = fc.import_mat(session_mat)
    df = pd.DataFrame(mat['sessionData'].T)
    if playback:
        print('playback')
        columns = columns + ['dt2']

    if len(df.columns) > len(columns):
        n_user_columns = len(df.columns) - len(columns)
        user_columns = [f'user{i}' for i in range(n_user_columns)]
        if playback:
            columns = columns[:-2] + user_columns + columns[-2:]
        else:
            columns = columns[:-1] + user_columns + columns[-1:]
    elif len(df.columns) < len(columns):
        raise ValueError('Number of column names should not exceed number of virmen channels.')

    df.columns = columns
    
    # Process heading
    df['h'] = fc.wrap_h(df['h_int']) * -1 # invert because want clockwise rotation to be positive (rightward)

    # 5V signal if lick
    df['lick'] = df['lick'] > 0.1 

    # Integrate dt to get timebase t, set to index
    df['t'] = df['dt'].cumsum()
    df.trial = df.trial.astype(int)
    if df['reward'].isna().any():
        df.loc[df['reward'].isna(), 'reward'] = 0
    df.reward = df.reward.astype(int)
    df.world_id = df.world_id.astype(int)
    df.inITI = df.inITI.astype(bool)
    df.lick = df.lick.astype(bool)

    return df


def load_as_vr(mouse=None, date=None, session='session_1', ops=None):
    if ops is None:
        ops = options.default_ops(imaging=False)
    path = define_path(mouse=mouse, date=date, session=session, ops=ops)
    return load_vr(path['virmen_mat'], columns=ops['virmen_mat_columns'])

def load_as_session(mouse=None, date=None, session='session_1', ops=None, recompute=False):
    if ops is None:
        ops = options.default_ops()
    path = define_path(mouse=mouse, date=date, session=session, ops=ops)
    if os.path.isfile(path['session_pickle']) and not recompute:
        return fc.load_pickle(path['session_pickle'])
    else:
        return Session(mouse=mouse, date=date, session=session, ops=ops)

def load_as_anndata(mouse=None, date=None, session='session_1', adata_filekey='adata_h5ad', ops=None, recompute=False, save=True, skip_if_not_saved=True):
    if ops is None:
        ops = options.default_ops()
    path = define_path(mouse=mouse, date=date, session=session, ops=ops)
    adata_file = path[adata_filekey]
    if os.path.isfile(adata_file) and not recompute:
        adata = anndata.read_h5ad(adata_file)
    else:
        if skip_if_not_saved:
            warnings.warn(f'No anndata file for {mouse} {date} {session}.')
            return None
        else:
            session = Session(mouse, date, session=session, ops=ops)
            adata = anndata.AnnData(X=session.X.values, obs=session.obs, var=session.var, uns=fc.to_h5ad_safe(session.uns), layers=session.layers)

            if save:
                adata.write(adata_file)
    return adata

def load_imaging_sessions(mouse, dates, load_fn=load_as_anndata, **kwargs):
    sessions = []
    for date in dates:
        key = mousedate_tokey(mouse, date)
        session = load_fn(**key, **kwargs)
        if session is not None:
            sessions.append(session)
            print(f'Loaded {mouse} session {date}.')
    return sessions

def load_vr_sessions(mouse, dates, **kwargs):
    sessions = []
    for date in dates:
        key = mousedate_tokey(mouse, date)
        session = load_as_vr(**key, **kwargs)
        maze = maze_id(define_path(**key, ops=options.default_ops(imaging=False)), strip=True)
        session.attrs = dict(**key, maze=maze)
        sessions.append(session)
        print(f'Loaded {mouse} session {date}.')
    return sessions

def load_vars(keys):
    var_ls = []
    for key in keys:
        path = define_path(**key)
        var = fc.load_pickle(path['var_pickle'])
        var['mouse'] = key['mouse']
        var_ls.append(var)
    var_cat = pd.concat(var_ls, axis=0)
    return var_cat

# Small functions
def intersect_idx(t1, t2):
    idx = (t1 >= t2.iloc[0]) & (t1 < t2.iloc[-1])
    return idx

def unpack_var_name(var_name):
    plane = int(var_name.split('plane')[1].split('_')[0])
    source = int(var_name.split('source')[-1])
    return plane, source

def fetch_stat(adata=None, path=None, plane=None):
    """
    Retrieve suite2p stats for all cells.
    """
    if adata is not None:
        path = adata.uns['path']
        md = adata.uns['metadata']
    else:
        assert path is not None
        md = get_metadata(path)
    if plane is None:
        stat = [np.load(path['stat_npy'].format(plane=plane), allow_pickle=True) for plane in range(md['nslices'])]
    else:
        stat = np.load(path['stat_npy'].format(plane=plane), allow_pickle=True)
    return stat

def fetch_cell_stat(adata=None, var_name=None, path=None, stat=None):
    """
    Retrieve suite2p stats for one cell.
    """
    plane, source = unpack_var_name(var_name)
    if stat is None:
        stat = fetch_stat(adata=adata, path=path, plane=plane)
    else:
        stat = stat[plane]
    
    stati = stat[source]
    return stati

def fetch_cell_stats(adata, stat_name):
    """
    Retrieve one suite2p stat for all cells.
    """
    stats = fetch_stat(adata=adata) # Prefetch all stats first, to speed up
    sr = pd.Series(index=adata.var_names, dtype='object')
    for var_name in adata.var_names:
        stati = fetch_cell_stat(adata=adata, var_name=var_name, stat=stats) # Index into prefetched stats
        if type(stat_name) is str:
            sr[var_name] = stati[stat_name]
        elif callable(stat_name):
            sr[var_name] = stat_name(stati)
        else:
            raise IOError('stat_name must be either string or function.')
    return sr

# Selecting cells
def source_isnotclipped(adata):
    bit_depth = int(adata.uns['metadata']['SI.hChannels.channelAdcResolution'].split(' ')[0][1:]) - 1
    max_val = 2**bit_depth * adata.uns['ops']['max_val']
    
    planes = range(adata.uns['metadata']['nslices'])
    Fs = [np.load(adata.uns['path']['F_npy'].format(plane=plane)) for plane in planes]
    isnotclipped = np.concatenate(list(map(lambda F: F.max(axis=1) < max_val, Fs)))
    print(f'Removing {(~isnotclipped).sum()} sources with peak value above {max_val}')
    return isnotclipped

def source_isnotnearedge(stat, mindist=10):
    """
    mindist in pixels.
    """
    isnotnearedge = np.array([stati['min_dist_to_edge'] > mindist for stati in stat])
    print(f'Removing {(~isnotnearedge).sum()} sources that are less than {mindist} pixels from the edge.')
    return isnotnearedge

def source_iscell(self):
    """
    Suite2p classifier output, iscell.npy columns are (iscell, probability).
    """
    planes = range(self.uns['metadata']['nslices'])
    iscell = np.concatenate([np.load(self.uns['path']['iscell_npy'].format(plane=plane)) for plane in planes])
    print(f'{(iscell[:, 0] == 0).sum()} sources classified as not cells.')
    return iscell[:, 0].astype(bool), iscell[:, 1]

def source_isredcell(self):
    """
    Suite2p red cell detection on channel 2, redcell.npy columns are (isredcell, probability).
    """
    planes = range(self.uns['metadata']['nslices'])
    redcell = np.concatenate([np.load(self.uns['path']['redcell_npy'].format(plane=plane), allow_pickle=True) for plane in planes])
    return redcell[:, 0].astype(bool), redcell[:, 1]

def select_cells(adata, criteria=['isnotclipped', 'isnotnearedge', 'iscell']):
    if 'iscell' in adata.var.columns:
        idx = np.ones(adata.n_vars, dtype=bool)
        for criterion in criteria:
            idx = idx & adata.var[criterion].to_numpy(dtype=bool)
        adata = adata[:, idx]
    adata = adata[:, ~np.isnan(adata.layers['dF']).any(axis=0)]
    return adata

def get_min_dist_to_edge(stati, Lx=512, Ly=512):
    min_dist_to_edge = min(stati['xpix'].min(), Lx - stati['xpix'].max(), 
                            stati['ypix'].min(), Ly - stati['ypix'].max())
    return min_dist_to_edge
# Generate event triggered frames
def export_mat_frame_index(idyx, t, mat_file):
    cellarray = np.empty(idyx.shape[1], dtype=object)
    for i, idy in enumerate(idyx.T):
        cellarray[i] = idy
    mat = {'frameIndex': cellarray, 't': t}

    os.makedirs(os.path.dirname(mat_file), exist_ok=True)
    scipy.io.savemat(mat_file, mat)
    
def export_mat_frames(idyx, t, session_mat, frames_mat):
    local_path = {
        'frameIndex_mat': frames_mat.split('.mat')[0] + '_index.mat',
        'session_mat': session_mat,
        'frames_mat': frames_mat
    }
    export_mat_frame_index(idyx, t, local_path['frameIndex_mat'])
    
    shell_script = """
    sbatch $HOME/code/preprocess_2p/eventTriggeredMovie.slurm {frameIndex_mat} {session_mat} {frames_mat}
    """.format(**local_path)
    print(shell_script)
    out = subprocess.call(shell_script, shell=True)

class Sync(object):
    
    def __init__(self, filename, metadata, ops={}):
        self.nslices = metadata['nslices']
        filename = str(filename)
        suffix = filename.split('.')[-1].lower()
        if suffix == 'abf':
            self.raw = fc.import_abf(filename)
        elif suffix == 'h5':
            self.raw = fc.import_h5(filename)
        elif suffix == 'edr':
            self.raw = fc.import_edr(filename)

        else:
            raise ValueError('File extension not supported.')

        self.raw = self.truncate(self.raw, ops)
        self.raw = self.rename_columns(self.raw)
        self.init_vr()
        self.init_scan(metadata)

    def truncate(self, raw, ops):
        if 'truncate_sync_s' in ops.keys():
            tlim = ops['truncate_sync_s']
            idt = (raw['t'] >= tlim[0]) & (raw['t'] < tlim[1])
            return raw[idt]
        else:
            return raw

    def rename_columns(self, raw):
        rename_columns = {
            'ballPitch': 'Ball_pitc',
            'ballRoll': 'Ball_roll',
            'ballYaw': 'Ball_yaw',
            'licks': 'Licks',
            'virmenClk': 'Virmen',
            'SIframeClk': 'ScanImage',
            'SIframeCl': 'ScanImage',
            'ScanImageTrigger': 'ScanImage',
            'Lick detection': 'Licks',
        }
        raw = raw.rename(rename_columns, axis=1)
        # Drop unconnected ground channels
        raw = raw.drop([col for col in raw.columns if col.startswith('Ground')], axis=1)
        return raw
        
    def init_vr(self):
        self.vr_idx = self.get_vr_idx()
        self.vr = self.raw.drop(['Virmen', 'ScanImage'], axis=1).iloc[self.vr_idx]

    def get_vr_idx(self, threshold=1.5):
        triggers_int = np.diff((self.raw['Virmen'] > threshold).astype(int))
        idx_start = np.where(triggers_int > 0)[0] + 1
        return idx_start
         
    def init_scan(self, metadata):
        if self.raw['ScanImage'].max() > 1:
            self.scan_idx = self.get_volume_idx(metadata)
        else:
            # If no ScanImage triggers
            self.scan_idx = self.get_volume_idx_no_scanimage_triggers(metadata)
        self.scan = self.raw.drop(['Virmen', 'ScanImage'], axis=1).iloc[self.scan_idx]
    
    def get_pockels_on(self, idx_start, idx_end, pockels_threshold=0.02):
        pockels = np.array(self.raw['Pockels'])
        pockels = pockels - pockels.min() # zero baseline - had issues with baseline going above 0.3
        pockels_on = np.zeros(len(idx_start))
        for i, (start, end) in enumerate(zip(idx_start, idx_end)):
            pockels_on[i] = pockels[start:end].mean() 
        pockels_on = pockels_on > pockels_threshold
        return pockels_on
    
    def get_frame_idx(self, metadata, threshold=2.5):
        triggers_int = np.diff((self.raw['ScanImage'] > threshold).astype(int))
        idx_start = np.where(triggers_int > 0)[0] + 1
        idx_end = np.where(triggers_int < 0)[0] + 1
        assert idx_start.shape == idx_end.shape

        # Remove incomplete frames (should be trailing)
        scan_frame_period = np.diff(np.vstack([idx_start, idx_end]), axis=0)[0]
        scan_frame_idx = scan_frame_period > (np.median(scan_frame_period) - 10)
        idx_start = idx_start[scan_frame_idx]
        idx_end = idx_end[scan_frame_idx]

        # Remove flyback frames, which come at the end of each volume
        nslices, nflyback = metadata['nslices'], metadata['nflyback']
        frames_pervol = nslices + nflyback
        if nflyback > 0:
            flyback = (np.arange(idx_start.size) % frames_pervol) >= nslices
            frame_idx = idx_start[~flyback]
        else:
            frame_idx = idx_start
        return frame_idx
        
    def get_volume_idx(self, metadata):
        # Compute volume acquisition metrics
        frame_idx = self.get_frame_idx(metadata)
        ntriggers = frame_idx.shape[0] # includes leftover triggers after abort
        nslices = metadata['nslices'] # flyback frames already removed in get_frame_idx
        nvolumes = int(np.floor(ntriggers / nslices))
        nframes = int(nvolumes * nslices)
        
        # Compute shift to select middle of volume acquisition
        sampling_rate = 1. / self.raw.t.iloc[:100].diff().mean()
        idx_shift = (nslices / 2) * (sampling_rate * metadata['SI.hRoiManager.scanFramePeriod'])
        
        # Subsample and shift frame_idx
        volume_idx = frame_idx[:nframes:int(nslices)] + int(idx_shift)   
        assert len(volume_idx) == nvolumes
        
        return volume_idx

    def get_volume_idx_no_scanimage_triggers(self, metadata, sigma_s=0.004):
        min_dist_s = 1 / metadata['SI.hRoiManager.scanVolumeRate']*0.9
        dt = self.raw['t'].diff().mean()
        min_dist = int(min_dist_s / dt)
        
        pockels = self.raw['Pockels'].values
        sigma = int(sigma_s / dt)
        pockels_filt = scipy.ndimage.gaussian_filter1d(pockels, sigma)
        thresh = pockels_filt.max() / 2
        pockels_thresh = pockels_filt > thresh
        
        start_inds = fc.rising_idx(pockels_thresh, min_dist=None, direction='rising')
        end_inds = fc.rising_idx(pockels_thresh, min_dist=None, direction='falling')
        assert len(start_inds) == len(end_inds)
        diff = (end_inds - start_inds)
        diff_mean = np.mean(diff)
        diff_std = np.std(diff)
        dev = np.abs(diff - diff_mean) 
        omit = dev > (diff_std * 6)
        assert omit.sum() <= 1
        start_inds = start_inds[~omit]
        end_inds = end_inds[~omit]
        scan_idx = ( (end_inds + start_inds) / 2).astype(int)
        return scan_idx
        

class Session(object):
    def __init__(self, mouse=None, date=None, session='session_1', ops=None):
        # Define path, metadata, parameters
        if ops is None:
            ops = options.default_ops()
        self.uns = {}
        self.uns['path'] = path = define_path(mouse=mouse, date=date, session=session, ops=ops)
        self.uns['metadata'] = metadata = get_metadata(path)
        ops['X_is_deconvolved'] = True
        self.uns['ops'] = ops

        # Load sync file
        t0 = time.perf_counter()
        self.sync = Sync(path['sync'], metadata, ops)
        self.uns['triggers'] = self._sync_triggers()
        # self.sync.scan = self.sync.scan.iloc[:-1] 
        # self.sync.scan_idx = self.sync.scan_idx[:-1]
        t1 = time.perf_counter()
        print('Time to load sync: %.2f s.' %(t1-t0))

        # Load Virmen file
        self.uns['vr'] = self._load_vr()
        self.uns['metadata']['maze'] = maze_id(self.uns['path'], strip=True)
        t2 = time.perf_counter()
        print('Time to load vr: %.2f s.' %(t2-t1))

        # Load suite2p footprint data
        self.stat = self._load_stat()

        # Load activity
        self.X = self._load_activity('spks')
        self.layers = {'dF': self._load_activity('dF').to_numpy(),
                        'dcnv': self.X}
        self._sync_activity()
        t3 = time.perf_counter()
        print('Time to load activity: %.2f s.' %(t3-t2))
            
        # Compute obs df
        self.obs = self._obs()
        t4 = time.perf_counter()
        print('Time to compute obs: %.2f s.' %(t4-t3))

        # Load suite2p mean images to get channel intensities
        self.img = self._load_mean_imgs()

        # Compute var df
        self.var = self._var(dilation=ops['dilation'])
        t5 = time.perf_counter()
        print('Time to load mean images and compute var: %.2f s.' %(t5-t4))

        # Extract and process photostim data
        if ops['is_photostim']:
            from mouse_imaging import photostimulation as ps
            ps.process_photostim_session(self)
            t5_2 = time.perf_counter()
            print('Time to compute photostimulation data: %.2f s.' %(t5_2-t5))
            t5 = t5_2

        # Compute maze-specific variables
        if ops['env']:
            env = importlib.import_module('mouse_imaging.' + ops['env'])
            env.main(self, maze=ops['maze'], do_median_trajectory=ops['do_median_trajectory'])
            

        t6 = time.perf_counter()
        print('Time to compute env variables: %.2f s.' %(t6-t5))

        fc.save_pickle(self, path['session_pickle'])
        t7 = time.perf_counter()
        print('Time to save pickle: %.2f s.' %(t7-t6))

        print(self)

    def __repr__(self):
        md = self.uns['metadata']
        string =  f"Session object for {md['mouse']} on {md['date']}, {md['session']}."
        return string


    def _sync_vr(self, vr, sync):
        vr['t'] = np.array(sync.vr.t.iloc[:len(vr)]) # Replace t with triggers from sync, trailing triggers are from trailing incomplete trial that is not saved
        vr['dt'] = vr['t'].diff()
        sync = self._subsample_sync(self.sync.vr_idx[:len(vr)])
        sync = sync.drop(['t', 'dt'], axis=1)
        assert len(vr) == len(sync)
        vr = pd.concat([vr, sync], axis=1)
        return vr

    def _load_vr(self):
        if os.path.isfile(self.uns['path']['virmen_mat']):
            vr_raw = load_vr(self.uns['path']['virmen_mat'], columns=self.uns['ops']['virmen_mat_columns'], playback=self.uns['ops']['is_vr_playback'])
            return self._sync_vr(vr_raw, self.sync)
        else:
            warnings.warn('No virmen sessionData.mat file found.')
            return

    def _load_activity(self, signal):
        activity_ls = []
        columns = []
        planes = range(self.uns['metadata']['nslices']) # 0-based suite2p planes, excludes flyback
        for plane in planes:
            if signal == 'dF':
                dF = preprocess_dF(path=self.uns['path'], ops=self.uns['ops'], fs=self.uns['metadata']['volume_rate'], plane=plane)
                activity_slicei = dF
            elif signal == 'spks':
                activity_slicei = np.load(self.uns['path']['spks_npy'].format(plane=plane)) # suite2p deconvolved activity
            columns.extend([f'plane{plane}_source{isource}' for isource in range(len(activity_slicei))])
            activity_ls.append(activity_slicei)
        
        # Remove trailing data from incomplete z-stack at end of recording
        min_rec_length = min([activity.shape[1] for activity in activity_ls])
        max_rec_length = max([activity.shape[1] for activity in activity_ls])
        assert (max_rec_length - min_rec_length) <= 1
        activity_ls = [activity_slicei[:, :min_rec_length] for activity_slicei in activity_ls]
        activity = np.concatenate(activity_ls, axis=0).T

        # Put in dataframe
        activity_df = pd.DataFrame(data=activity, columns=columns)
        activity_df.index = activity_df.index.astype(str) # otherwise AnnData throws warning
        return activity_df

    def _sync_activity(self):
        if self.uns['vr'] is not None:
            scan2vr_idx = intersect_idx(self.sync.scan['t'], self.uns['vr']['t']).to_numpy()
        else:
            scan2vr_idx = np.ones(len(self.sync.scan)).astype(bool)

        # Check if frames number is different in sync vs tif
        assert len(self.X) == len(self.layers['dF'])
        scan_tif_frame_diff = len(scan2vr_idx) - len(self.X)
        if np.abs(scan_tif_frame_diff)> 0:
            print(f'{scan_tif_frame_diff} frames difference between ScanImage triggers and tif files.')

            # Equalize frames between sync and tif
            if len(self.X) < len(scan2vr_idx):
                scan2vr_idx = scan2vr_idx[:len(self.X)]
                self.sync.scan = self.sync.scan.iloc[:len(self.X)]
                self.sync.scan_idx = self.sync.scan_idx[:len(self.X)]
            elif len(self.X) > len(scan2vr_idx):
                self.X = self.X[:len(scan2vr_idx)]
                for layer in self.layers.keys():
                    self.layers[layer] = self.layers[layer][:len(scan2vr_idx)]

        # Select imaging frames that overlap with VR
        self.X = self.X[scan2vr_idx]
        for layer in self.layers.keys():
            self.layers[layer] = self.layers[layer][scan2vr_idx]
        self.scan2vr_idx =  self.uns['triggers']['scan2vr_idx'] = scan2vr_idx

    def _load_stat(self, xmargin=10, ymargin=10):
        Ly, Lx = self.uns['metadata']['Ly'], self.uns['metadata']['Lx']
        stat_ls = []
        planes = range(self.uns['metadata']['nslices'])
        for plane in planes:
            stat_slicei = np.load(self.uns['path']['stat_npy'].format(plane=plane), allow_pickle=True)
            for stati in stat_slicei: 
                stati['slice'] = plane # 0-based

                # Compute min distance to edge
                min_dist_to_edge = min(stati['xpix'].min(), Lx - stati['xpix'].max(), 
                                        stati['ypix'].min(), Ly - stati['ypix'].max())
                stati['min_dist_to_edge'] = min_dist_to_edge
            stat_ls.append( stat_slicei )
        stat = np.concatenate(stat_ls, axis=0)
        return stat

    def _subsample_vr(self, interpolation_kind='nearest', max_gap_s=0.5):
        if (self.uns['vr'].dt > max_gap_s).any():
            warnings.warn(f'A gap of %.2f s occured in Virmen timestamps.' %self.uns['vr'].dt.max())

        def cast(y1, t1, t2):
            assert len(t1) == len(y1)
            interpolate = scipy.interpolate.interp1d(t1, y1, kind=interpolation_kind)
            return interpolate(t2)
        
        # Subsample vr to match scan timebase
        t_vr = self.uns['vr'].t.copy()
        t_scan = self.sync.scan.t[self.scan2vr_idx].copy()
        vr = self.uns['vr'].drop(['t', 'dt'], axis=1)
        vr_subsampled = vr.apply(lambda col: cast(col, t_vr, t_scan), axis='index')

        # Commented out because now do this in _subsample_sync
        # vr_subsampled['t'] = np.array(t_scan)
        # vr_subsampled['dt'] = vr_subsampled['t'].diff()
        
        # Set data types
        vr_subsampled.trial = vr_subsampled.trial.astype(int)
        if vr_subsampled['reward'].isna().any():
            warnings.warn(f'NaNs in reward channel.')
            vr_subsampled.loc[vr_subsampled['reward'].isna(), 'reward'] = 0
        vr_subsampled['reward'] = vr_subsampled['reward'].astype(int)
        vr_subsampled.world_id = vr_subsampled.world_id.astype(int)
        vr_subsampled.inITI = vr_subsampled.inITI.astype(int)
        vr_subsampled.lick = vr_subsampled.lick.astype(int)

        return vr_subsampled

    def _subsample_sync(self, idt):
        sync_subsampled = pd.DataFrame()
        for sync_label, obs_label in self.uns['ops']['sync_labels'].items():
            if sync_label in self.sync.raw.keys():
                signal = np.array(self.sync.raw[sync_label].iloc[idt])
                offset = self.uns['ops']['sync_offsets'][sync_label]
                gain = self.uns['ops']['sync_gains'][sync_label]
                sync_subsampled[obs_label] = (signal - offset) * gain
            else:
                warnings.warn(f'Channel "{sync_label}" not in sync file.')

        # Add time
        sync_subsampled['t'] = np.array(self.sync.raw['t'].iloc[idt])
        sync_subsampled['dt'] = sync_subsampled['t'].diff()

        # sync_subsampled['sync_reward'] = sync_subsampled['sync_reward'] > 1
        return sync_subsampled

    def _obs(self, interpolation_kind='nearest', max_gap_s=0.5):
        # Subsample sync data to scan timebase
        obs = self._subsample_sync(self.sync.scan_idx[self.scan2vr_idx])

        if os.path.isfile(self.uns['path']['virmen_mat']):
            # Subsample vr data to scan timebase
            vr_subsampled = self._subsample_vr(interpolation_kind=interpolation_kind, max_gap_s=max_gap_s)
            overlapping_columns = set(obs.columns).intersection(vr_subsampled.columns)
            obs = pd.concat([obs, vr_subsampled.drop(overlapping_columns, axis=1)], axis=1)

        obs.index = obs.index.astype(str) # otherwise AnnData throws warning
        obs = obs[:len(self.X)]
        return obs

    def _load_mean_imgs(self):
        """
        Registered mean image of each channel from suite2p ops.npy, shape (nplanes, nchannels, Ly, Lx).
        Channel order follows the filter in the tif filename, e.g. G1R1 -> ['G', 'R'], with the functional channel as suite2p channel 1.
        """
        md = self.uns['metadata']
        mean_img_keys = ['meanImg', 'meanImg_chan2'][:md['nchannels']]
        img = np.zeros((md['nslices'], md['nchannels'], md['Ly'], md['Lx']))
        for plane in range(md['nslices']):
            ops_plane = np.load(self.uns['path']['ops_npy'].format(plane=plane), allow_pickle=True).item()
            for ichannel, key in enumerate(mean_img_keys):
                img[plane, ichannel] = ops_plane[key]
        return img

    def get_img(self, channel, plane, filter_key=None):
        """
        Mean image of one channel ('G', 'R', ...) in one 0-based plane. filter_key is unused, kept for compatibility.
        """
        channels = self.uns['metadata']['filter1']['channels']
        assert channel in channels # check channel is present
        img = self.img[plane, channels.index(channel)]

        if (img == 0).all():
            warnings.warn('Requested image is blank.')

        return img

    def _var(self, dilation=None):
        var = pd.DataFrame(index=self.X.columns)

        # Compute intensity for each channel
        for channel in self.uns['metadata']['available_channels']:
            var[channel] = self._cell_means(channel, **dilation)

        var['isnotclipped'] = source_isnotclipped(self)
        var['isnotnearedge'] = source_isnotnearedge(self.stat, mindist=self.uns['ops']['min_dist_to_edge'])
        var['iscell'], var['iscell_prob'] = source_iscell(self)
        if self.uns['metadata']['nchannels'] > 1:
            var['redcell'], var['redcell_prob'] = source_isredcell(self)
        var['plane'] = [stati['slice'] for stati in self.stat]

        # Add median cell coordinates
        assert len(self.stat) == len(var)
        var['y'], var['x'] = np.stack([stati['med'] for stati in self.stat]).T.astype(int)

        return var

    def _sync_triggers(self):
        triggers = {}
        if 'triggers' in self.uns['ops'].keys():
            for trig_name in self.uns['ops']['triggers']:
                if trig_name in self.sync.raw.columns:
                    triggers[trig_name] = self.sync.raw['t'].iloc[fc.rising_idx(self.sync.raw[trig_name])].to_numpy()
                else:
                    warnings.warn(f'Trigger "{trig_name}" not in sync file.')
        return triggers

    def _cell_means(self, channel, cell_dilation=0, background_dilation=None, filter_key=None):  
        intensity = pd.Series(np.nan, index=self.X.columns)
        for icell in range(len(self.stat)):
            cell = self._cell_mean(icell, channel, dilation=cell_dilation)
            if background_dilation:
                background = self._cell_mean(icell, channel, shell=background_dilation, filter_key=filter_key)
                cell = cell - background
            intensity.iloc[icell] = cell
        return intensity

    def _cell_mean(self, icell, channel, dilation=0, shell=None, filter_key=None):
        img = self.get_img(channel, self.stat[icell]['slice'], filter_key=filter_key)
        roi = self._cell_mask(icell, dilation=dilation, shell=shell)
        intensity = img[roi].mean()
        return intensity

    def _cell_mask(self, icell, dilation=0, shell=None):
        stati = self.stat[icell]
        mask = np.zeros((self.uns['metadata']['Ly'], self.uns['metadata']['Lx'])).astype(bool)
        mask[stati['ypix'], stati['xpix']] = True
        if dilation:
            mask = fc.dilate(mask, dilation=dilation)
        elif shell:
            outer_dilation, inner_dilation = shell
            outer = fc.dilate(mask, dilation=outer_dilation)
            inner = fc.dilate(mask, dilation=inner_dilation)
            mask = outer
            mask[inner] = False
        return mask

    def _cell_img(self, icell, channel, crop=(60, 60), filter_key=None):
        Ly, Lx = self.uns['metadata']['Ly'], self.uns['metadata']['Lx']
        stati = self.stat[icell]
        img = self.get_img(channel, stati['slice'], filter_key=filter_key)
        y, x = map(int, stati['med'])
        
        dx, dy = map(lambda x: int(x/2), crop)
        top = max(0, y-dy)
        bottom = min(Ly, y+dy)
        left = max(0, x-dx)
        right = min(Lx, x+dx)
        cell_img = img[top:bottom, left:right]
        return cell_img

    def _cell_outline(self, icell, crop=(60, 60), shell=(1, 0)):
        Ly, Lx = self.uns['metadata']['Ly'], self.uns['metadata']['Lx']
        stati = self.stat[icell]
        outline = self._cell_mask(icell, shell=shell)
        y, x = map(int, stati['med'])
        dx, dy = map(lambda x: int(x/2), crop)
        top = max(0, y-dy)
        bottom = min(Ly, y+dy)
        left = max(0, x-dx)
        right = min(Lx, x+dx)
        cell_outline = outline[top:bottom, left:right]
        return cell_outline

    def imshow_cell(self, icell, channels='all', crop=(60, 60), filter_key=None, title=None):
        if channels == 'all':
            channels = self.uns['metadata']['available_channels']
        fig = plt.figure(figsize=(len(channels)*5, 5))
        if title:
            fig.suptitle(title)
        gs = mpl.gridspec.GridSpec(ncols=len(channels), nrows=1, figure=fig)
        for ichannel, channel in enumerate(channels):
            img = self._cell_img(icell, channel, crop=crop, filter_key=filter_key)
            outline = self._cell_outline(icell, crop=crop, shell=(1, 0))
            ax = fig.add_subplot(gs[0, ichannel])
            plt.imshow(img)
            plt.colorbar(ax=ax, shrink=0.5)
            plt.imshow(outline, alpha=0.1)
            ax.set_title(channel)

    def save(self, light=True):
        if light:
            obj = copy.deepcopy(self)
            delattr(obj, 'sync')
            filename = 'session_pickle'
        else:
            obj = self
            filename = 'session.wsync.pickle'
        fc.save_pickle(obj, os.path.join(self.uns['path']['preprocessed_dir'], filename))

def main(mouse, date, session='session_1', ops=None, recompute=True, save=True):
    if ops is None:
        ops = options.default_ops()
    adata = load_as_anndata(mouse=mouse, date=date, session=session, ops=ops, adata_filekey='adata_allsources_h5ad', recompute=recompute, save=False, skip_if_not_saved=False)
    adata = select_cells(adata)
    t0 = time.perf_counter()

    if 'process_fcn' in ops.keys():
        process = getattr(options, ops['process_fcn'])
        ops = adata.uns['ops']
        adata = process(adata, do_corrmat=ops['do_corrmat'], do_nneighbor_graph=ops['do_nneighbor_graph'], do_umap=ops['do_umap'], do_leiden=ops['do_leiden'])

    # Compute average photostim frames
    # if adata.uns['ops']['is_photostim']:
    #     ps.export_photostim_average_frames(adata, t_range=(-2, 5))

    save_adata(adata)
    t1 = time.perf_counter()
    print('Time to run adata processing: %.2f' %(t1-t0))
    
    return adata

def update_function(adata):
    print("Processing {mouse} {date} {session}.".format(**adata.uns['metadata']))
    from mouse_imaging import analysis as an
    from mouse_imaging import photostimulation as ps
    importlib.reload(an)
    importlib.reload(ps)
    from sklearn import linear_model
    update_path(adata)

    def add_influence_metrics(adata, min_dist_um=40):
        influence_metric = 'influence_0.5-1.0s_%ium-mindist' %min_dist_um
        influence_value, pvalue, ntrials_stim, ntrials_ctl = ps.compute_influence_value(
                    adata, stim_group=1, ctl_group=2, tlim0=(-1, 0), tlim=(0.5, 1.0),
                    min_dist_um=min_dist_um, max_trials=200)
        adata.var[influence_metric + '_value'] = influence_value
        adata.var[influence_metric + '_pvalue'] = pvalue
        adata.var['ntrials_stim'] = ntrials_stim
        adata.var['ntrials_ctl'] = ntrials_ctl

        activity_value = ps.compute_activity_value(adata, ctl_group=2, max_trials=200, 
                                           tlim=(0.5, 1.0), tlim0=(-1, 0), layer='dcnv_norm')
        adata.var['activity_diff_in_influence_window'] = activity_value

        influence_metric = 'influence_0.5-1.0s_%ium-mindist_nodiff' %min_dist_um
        influence_value, pvalue, ntrials_stim, ntrials_ctl = ps.compute_influence_value(
                    adata, stim_group=1, ctl_group=2, tlim0=None, tlim=(0.5, 1.0),
                    min_dist_um=min_dist_um, max_trials=200)
        adata.var[influence_metric + '_value'] = influence_value
        adata.var[influence_metric + '_pvalue'] = pvalue
        adata.var['ntrials_stim_nodiff'] = ntrials_stim
        adata.var['ntrials_ctl_nodiff'] = ntrials_ctl

        activity_value = ps.compute_activity_value(adata, ctl_group=2, max_trials=200, 
                                               tlim=(0.5, 1.0), tlim0=None, layer='dcnv_norm')
        adata.var['activity_in_influence_window'] = activity_value
                
        adata.var['dist_to_targets_center_of_mass'] = ps.dist_to_targets_center_of_mass(adata, stim_group=1)
        adata.var[f'dist_to_closest_sigstim_target_after_{min_dist_um}um'] = ps.dist_to_closest_target(adata, group=1, min_dist_um=min_dist_um, require_significant_stim=True)
        adata.var[f'dist_to_closest_target_after_{min_dist_um}um'] = ps.dist_to_closest_target(adata, group=1, min_dist_um=min_dist_um, require_significant_stim=False)

    def linregress_stats(adata, obs_col, obs_key=None):
        stats = pd.DataFrame(index=adata.var_names, columns=['slope', 'rvalue', 'pvalue'], dtype=np.float64)
        obs_idx = an.fetch_index(adata.obs, obs_key)
        x = scipy.stats.zscore(adata.obs.loc[obs_idx, obs_col])
        Y = adata[obs_idx, :].layers['dcnv_norm']
        for var_name, y in zip(adata.var_names, Y.T):
            reg = scipy.stats.linregress(x, y)
            stats.loc[var_name, 'slope'] = reg.slope
            stats.loc[var_name, 'rvalue'] = reg.rvalue
            stats.loc[var_name, 'pvalue'], reg.pvalue
        return stats

    def add_variables(adata):
        adata.obs['stim_on_2s'] = ps.photostim_idx(adata.obs, adata, after_s=2, stim_group=1)
        adata.obs['abs_h_mt_error'] = np.abs(adata.obs['h_mt_error'])
        adata.obs['abs_ddh_0.25sigma'] = scipy.ndimage.gaussian_filter1d(np.abs(adata.obs['ddh']), sigma=0.25/adata.obs['dt'].mean())
        adata.obs['h_mt_error*ddh_0.25sigma'] = adata.obs['h_mt_error'] * adata.obs['ddh_0.25sigma']
        adata.obs['pitch_0.25sigma'] = scipy.ndimage.gaussian_filter1d(adata.obs['pitch'], sigma=0.25/adata.obs['dt'].mean())
    
    
    # Add influence metrics
    add_influence_metrics(adata, min_dist_um=40)

    # Add variables
    add_variables(adata)
    
    # Cell tuning
    obs_key = {'stim_on_2s': False,
                'inITI': False,
                'correct': True}
    adata.var['h_mt_error_max'] = an.cell_tuning(adata, obs_col='h_mt_error', obs_key=obs_key, bins=np.arange(-np.pi, np.pi+0.01, np.pi/8),)
    adata.var['abs_h_mt_error_max'] = an.cell_tuning(adata, obs_col='abs_h_mt_error', obs_key=obs_key, bins=np.arange(0, np.pi+0.01, np.pi/8),)
    adata.var['y_max'] = an.cell_tuning(adata, obs_col='y', obs_key=obs_key, bins=np.arange(0, 331, 30),)

    # Multiple linear regression
    predictors = ['pitch_0.25sigma', 'abs_ddh_0.25sigma', 'abs_h_mt_error', 'h_mt_error*ddh_0.25sigma']
    var_cols = [predictor + '_coef' for predictor in predictors]
    adata.var[var_cols] = an.activity_regression(adata, predictors, model=linear_model.LinearRegression(), obs_key=obs_key)

    # Linear regression on each predictor separately
    linregress_metrics = ['slope', 'rvalue', 'pvalue']
    for predictor in predictors:
        var_cols = [f'{predictor}_' + metric for metric in linregress_metrics]
        stats = linregress_stats(adata, predictor, obs_key=obs_key)
        adata.var[var_cols] = stats[linregress_metrics]

    fc.save_pickle(adata.var, adata.uns['path']['var_pickle'])

def update_adata(mouse, date, session='session_1'):
    # adata_filekey = 'adata_h5ad'
    adata = load_as_anndata(mouse=mouse, date=date, session=session, adata_filekey='adata_h5ad', recompute=False, save=False)
    # adata.write(adata.uns['path']['adata_h5ad'] + '.backup')
    update_function(adata)
    save_adata(adata, adata_filekey='adata_h5ad')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Load session as anndata and save pickle.')
    parser.add_argument('--mouse', required=True, type=str)
    parser.add_argument('--date', required=True, type=str)
    parser.add_argument('--session', required=False, type=str, default='session_1')
    parser.add_argument('--ops', required=False, type=str, default='default_ops')
    parser.add_argument('--update', action='store_true')
    args = vars(parser.parse_args())
    do_update = args.pop('update')
    if do_update:
        args.pop('ops')
        update_adata(**args)
    else:
        args['ops'] = getattr(options, args['ops'])()
        main(**args)
