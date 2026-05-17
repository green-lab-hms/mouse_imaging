import os, importlib, warnings
import scipy
from mouse_imaging import *
import numpy as np
import importlib
importlib.reload(sess)
def format_date(line):
    month, day, year = line.strip().split('/')
    date = ''.join([year[-2:], month.zfill(2), day.zfill(2)])
    return date

def textfile_to_keys(txt_file, date_range=None):
    mouse = os.path.basename(txt_file).split('.')[0]
    session = 'session_1'
    with open(txt_file) as f:
        lines = f.readlines()
    keys = [dict(mouse=mouse, date=format_date(line), session='session_1') for line in lines]
    if date_range is not None:
        if date_range[0] is not None:
            keys = [key for key in keys if key['date'] >= date_range[0]]
        if date_range[1] is not None:
            keys = [key for key in keys if key['date'] <= date_range[1]]
    return keys

def load_preprocessed_vr(mat_path, ops, subsample_dt=None):
    vr = sess.load_vr(mat_path, ops['virmen_mat_columns'])
    if subsample_dt is not None:
        vr = subsample_vr(vr, subsample_dt=subsample_dt, interpolation_kind='nearest')
    world_dict = tmaze.worlds(ops['maze'])
    tmaze.preprocess_vr(vr)
    trials = tmaze.compute_trials_df(vr, world_dict)
    tmaze.compute_vr_maze_variables(vr, trials, world_dict)
    return vr

def subsample_vr(vr, subsample_t=None, subsample_dt=None, interpolation_kind='nearest', max_gap_s=0.5):
    if (vr['dt'] > max_gap_s).any():
        warnings.warn(f'A gap of %.2f s occured in Virmen timestamps.' %vr['dt'].max())
    
    if subsample_t is None:
        subsample_t = np.arange(vr['t'].iloc[0], vr['t'].iloc[-1], subsample_dt)
    
    def cast(y1, t1, t2):
        assert len(t1) == len(y1)
        interpolate = scipy.interpolate.interp1d(t1, y1, kind=interpolation_kind)
        return interpolate(t2)

    # Subsample vr to match scan timebase
    t = vr['t'].copy()
    vr = vr.drop(['t', 'dt'], axis=1)
    vr_subsampled = vr.apply(lambda col: cast(col, t, subsample_t), axis='index')
    
    # Commented out because now do this in _subsample_sync
    vr_subsampled['t'] = subsample_t
    vr_subsampled['dt'] = vr_subsampled['t'].diff()

    # Set data types
    vr_subsampled.trial = vr_subsampled.trial.astype(int)
    vr_subsampled.reward = vr_subsampled.reward.astype(int)
    vr_subsampled.world_id = vr_subsampled.world_id.astype(int)
    vr_subsampled.inITI = vr_subsampled.inITI.astype(int)
    vr_subsampled.lick = vr_subsampled.lick.astype(int)
    return vr_subsampled

def long_short_trial_diff_hist2d(key, x=None, y=None, xbins=None, ybins=None, ops=options.default_ops(imaging=False), subsample_dt=None, world=None, low=50, high=50):
    path = sess.define_path(**key, ops=ops)
    if not os.path.isfile(path['virmen_mat']):
        print('Could not find a file for {mouse} {date} {session}.'.format(**key))
        return
    else:
        print('Processing data for {mouse} {date} {session}.'.format(**key))
    vr = load_preprocessed_vr(path['virmen_mat'], ops, subsample_dt=subsample_dt)

    trial_len_s = vr.groupby('trial').apply(lambda x: len(x[x['inITI']==0]))*vr['dt'].mean()
    vr = vr[~vr.isna().any(axis=1)]
    lo_thresh = np.percentile(trial_len_s, low)
    hi_thresh = np.percentile(trial_len_s, high)
    idx = vr['correct'] & (vr['world'] == world) & ~vr['inITI'] & (vr['trial_len_s'] < 60)
    idx0 = idx & (vr['trial_len_s'] < lo_thresh)
    idx1 = idx & (vr['trial_len_s'] > hi_thresh)
    
    H0, x_edges, y_edges = np.histogram2d(vr.loc[idx0, x], vr.loc[idx0, y], bins=[xbins, ybins], density=True,)
    H1, x_edges, y_edges = np.histogram2d(vr.loc[idx1, x], vr.loc[idx1, y], bins=[xbins, ybins], density=True,)

    maze = sess.maze_id(path, strip=True, load_if_not_saved=True)
    data = dict(**key, maze=maze, H=H1-H0, w=idx.sum(), x_edges=x_edges, y_edges=y_edges,
               x=x, y=y, xbins=xbins, ybins=ybins, ops=ops, subsample_dt=subsample_dt, 
                world=world, low=low, high=high)
    return data

def compute_binned_statistic(key, x=None, y=None, values=None, xbins=None, ybins=None, region=None, max_trial_len_s=np.inf, obs_key=None, min_stderr=0.3):
    path = sess.define_path(**key)
    if os.path.isfile(path['adata_h5ad']):
        print('Processing data for {mouse} {date} {session}.'.format(**key))
        adata = sess.load_as_anndata(**key)
    else:
        print('Could not find a file for {mouse} {date} {session}.'.format(**key))
        return
    assert 'dF' in adata.layers.keys()
    if region is not None:
        assert adata.uns['metadata']['region'] == region
    
    obs = adata.obs
    obs = obs[(~obs['inITI']) & (obs['trial_len_s']<max_trial_len_s)]
    obs = obs[~obs[[x, y, values]].isna().any(axis=1)]
    obs = obs[an.fetch_index(obs, obs_key)]
    
    def bootstrap_stderr(arr1d,):
        assert len(arr1d.shape)==1
        arr1d = arr1d[~np.isnan(arr1d)]
        if len(arr1d) < 3:
            return np.inf
        else:
            bootstrap_result = scipy.stats.bootstrap((arr1d,), statistic=np.nanmean, confidence_level=0.95, n_resamples=20,)
            return bootstrap_result.standard_error

    H, x_edges, y_edges, binnumber = scipy.stats.binned_statistic_2d(x=obs[x], y=obs[y], values=obs[values], statistic=np.nanmean, bins=[xbins, ybins],)
    
    if min_stderr > 0:
        # Mask bins where stderr > range * min_stderr
        stderr, _, _, _ = scipy.stats.binned_statistic_2d(x=obs[x], y=obs[y], values=obs[values], statistic=bootstrap_stderr, bins=[xbins, ybins],)
        H[stderr > (H*min_stderr)] = np.nan
    
    data = dict(**key, H=H, w=len(obs), x_edges=x_edges, y_edges=y_edges,
               x=x, y=y, values=values, xbins=xbins, ybins=ybins, region=region, max_trial_len_s=max_trial_len_s, obs_key=obs_key, min_stderr=min_stderr)

    return data

def compile_hist2d(keys, compile_fcn, **kwargs):
    data = []
    for key in keys:
        datai = compile_fcn(key, **kwargs)
        if datai is not None:
            data.append(datai)
    
    data2 = data[0].copy()
    data2['H'] = np.stack([datai['H'] for datai in data])
    data2['w'] = np.array([datai['w'] for datai in data])
    data2['date'] = [datai['date'] for datai in data]
    data2['session'] = [datai['session'] for datai in data]
    data2['maze'] = [datai['maze'] for datai in data]
    return data2

def main(mouse=None, date_range=None, save=True):
    dates_file = f'/n/data2/hms/neurobio/harvey/jonathan/analysis/imaging/virmen_dates/{mouse}.txt'
    keys = textfile_to_keys(dates_file, date_range=date_range)
    kwargs = dict(
        x='h_error', 
        y='ddh', 
        xbins=np.arange(-np.pi, np.pi+0.001, 0.15), 
        ybins=np.arange(-1, 1+0.001, 0.05), 
        world='white_right',
        subsample_dt=1/7.5)
    data = compile_hist2d(keys, compile_fcn=long_short_trial_diff_hist2d, **kwargs)
    if save:
        fc.save_pickle(data, f'/n/data2/hms/neurobio/harvey/jonathan/analysis/imaging/SstCre-RFP_Sst44-nlsBFP/compiled_behavior_data/long_short_trial_diff_hist2d/{mouse}.pickle')
    return data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fluorescence trace processing into deconvolved signal.')
    parser.add_argument('--mouse', required=True, type=str)
    args = vars(parser.parse_args())
    main(**args)