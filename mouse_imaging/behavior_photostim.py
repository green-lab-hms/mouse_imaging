import os, glob
import pandas as pd
from mouse_imaging import *
from mouse_imaging import functions as fc

import scipy.io
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import importlib

importlib.reload(tmaze)

# Plot
def plot_accuracy(trials, ylim=(0, 1), key={}, x='h_pulse', y='correct', hue='stim_location', print_sample_size=True):
    if type(trials) is list:
        trials = pd.concat(trials, axis=0)
    plt.figure(figsize=(0.75, 1))
    # plt.suptitle(f"{mouse} {date}")
    ax = plt.subplot()

    idx = an.fetch_index(trials, key)
    palette = {'Laser ON PPC': 'tab:blue', 'Laser ON Control': 'lightblue', 'Laser OFF': 'silver'}
    sns.barplot(data=trials[idx], x=x, y=y, hue=hue, errwidth=0.5, palette=palette)
    plt.ylim(ylim)
    pl.format_plot(ax, remove_spines=['top', 'right'])
    plt.legend(bbox_to_anchor=(1.05, 1))
    
    if print_sample_size:
        # print(trials[idx].groupby([x]).size())
        # print(trials[idx].groupby([hue]).size())
        print(trials[idx].groupby([x, hue]).size())

def plot_trigger_1d(vrs, obs_keys_set=None, offset_s=0, figsize=(0.6, 2.5), **kwargs):
    lw = 0.5
    plot_kwargs = [
        dict(col='h_bias_radpersec', ylim=(-2, 2), yticks=(-2, 0, 2), hline=0, lw=lw),
        dict(col='target_dist', ylim=(0, 300), yticks=(0, 100, 200, 300), lw=lw),
        # dict(col='h_mt_error', ylim=(-np.pi, 0, np.pi), 
        #      yticks=[-np.pi, 0, np.pi], hline=0, statistic=fc.nancircmean, lw=lw),
        dict(col='abs_h_mt_error', ylim=(0, np.pi), 
             yticks=[0, np.pi], hline=0, statistic=np.nanmean, lw=lw),
        dict(col='yaw_corr', ylim=(-1, 1), yticks=[-1, 0, 1], hline=0, lw=lw),
        dict(col='dyaw_corr', ylim=(-2, 2), yticks=[-2, 0, 2], hline=0, lw=lw),
    ]
    trig_kwargs = dict(trigger_name='y>150_first', min_dist_s=0, offset_s=offset_s, t_range=(-2, 10))
    # Plot
    plt.figure(figsize=figsize)
    # plt.suptitle(pl.sessions_title(adatas), y=1.15)
    overlay_kwargs = [dict(color='tab:blue'), dict(color='lightblue')]
    pl.subgrid_trig_1d(vrs, trig_kwargs=trig_kwargs, obs_keys_set=obs_keys_set, 
                       plot_kwargs=plot_kwargs, overlay_kwargs=overlay_kwargs,
                       **kwargs)
# Analysis
import scipy.ndimage
stimid_rename = {'na': 'Laser OFF', 'control': 'Laser ON Control', 'PPCadj': 'Laser ON PPC'}
stimid_map = {'Laser OFF': 0, 'Laser ON Control': 1, 'Laser ON PPC': 2}
stimid_map_inv = {v: k for k, v in stimid_map.items()}

def first_crossing_ind(triali):
    inds = fc.rising_idx(triali, direction='rising')
    if len(inds) == 0:
        return len(triali)-1
    else:
        return inds[0]
    
def y_first_idx(vr, y):
    vr = vr.copy()
    vr['tmp'] = vr['y'] > y
    trial_start_inds = np.insert(fc.rising_idx(vr['trial']), 0, 0)
    trial_inds = vr.groupby('trial')['tmp'].apply(first_crossing_ind)
    first_inds = trial_start_inds + trial_inds
    first_idx = np.zeros(len(vr)).astype(bool)
    first_idx[first_inds] = True
    return first_idx

def add_stim_trial(vr):
    trials = vr.attrs['trials']
    # Check that trial vars have been defined
    assert 'stim_location_num' in trials.columns

    trial_len = vr.groupby('trial').apply(lambda x: len(x))
    repeat_cols = ['photostim_on', 'photostim_on_prev', 'stim_location', 'stim_location_prev']
    for col in repeat_cols:
        vr[col + '_trial'] = tmaze.repeat_trial_var(trials[col], trial_len)

def correct_yaw(obs):
    obs['yaw_corr'] = obs['yaw'].copy()
    obs_nopulse = obs[obs['h_pulse_trial']==False]
    result = scipy.stats.linregress(x=obs_nopulse['yaw'], y=obs_nopulse['dh'])
    obs['yaw_corr'] = obs['yaw'] * result.slope + result.intercept
    obs['dyaw_corr'] = np.gradient(obs['yaw_corr'], obs['t'])

def add_vr_columns(obs, y_pulse):
    obs[f'y>{y_pulse}_first'] = y_first_idx(obs, y=y_pulse)

    obs['dyaw_corr_0.25sigma'] = scipy.ndimage.gaussian_filter1d(obs['dyaw_corr'], sigma=0.25/obs['dt'].mean())
    obs['dyaw_corr_0.25sigma>0.15'] = obs['dyaw_corr_0.25sigma'] > 0.15
    obs['dyaw_corr_0.25sigma<-0.15'] = obs['dyaw_corr_0.25sigma'] < -0.15
    obs['h_bias_radpersec'] = obs['h_bias'] * 60
    obs['abs_dyaw_corr_0.25sigma<0.1'] = np.abs(obs['dyaw_corr_0.25sigma']) < 0.1
    
    obs['abs_h_mt_error'] = np.abs(obs['h_mt_error'])
    
    for h_signal in ['h',]: # 'h_mt_error'
        obs[f'abs_{h_signal}<pi/3'] = np.abs(obs[h_signal]) < np.pi/3
        obs[f'abs_{h_signal}>pi/3'] = np.abs(obs[h_signal]) >= np.pi/3
    
        delay_s = 2
        delay = int(delay_s / obs['dt'].mean())
        for col in [f'abs_{h_signal}<pi/3', f'abs_{h_signal}>pi/3',]:
            obs[col + f'_+{delay_s}s'] = fc.delay_signal(obs[col], delay=delay)

    
    obs['abs_yaw_corr<0.5'] = np.abs(obs['yaw_corr']) < 0.5
    obs['yaw_corr>0.5'] = obs['yaw_corr'] > 0.5
    obs['yaw_corr<-0.5'] = obs['yaw_corr'] < -0.5
    
    obs['dh>0.5'] = obs['dh'] > 0.5
    obs['dh<-0.5'] = obs['dh'] < -0.5
    obs['abs_dh<0.5'] = np.abs(obs['dh']) < 0.5
    
def add_vr_variables(vr, y_pulse=150,):
    print('Adding vr variables to {mouse} {date} {session}.'.format(**vr.attrs))
    
    add_stim_trial(vr)
    correct_yaw(vr)
    add_vr_columns(vr, y_pulse=y_pulse)

def add_trial_variables(vr):
    print('Adding trial variables to {mouse} {date} {session}.'.format(**vr.attrs))
    trials = vr.attrs['trials']
    
    # Pulse trials
    trials['h_pulse'] = vr.groupby('trial')['h_pulse'].max()
    trials['h_pulse_left'] = vr.groupby('trial')['h_pulse_left'].max()
    trials['h_pulse_right'] = vr.groupby('trial')['h_pulse_right'].max()
    
    # Photostim trials
    trials['photostim_on'] = vr.groupby('trial')['photostim_on'].max()
    trials['photostim_on_prev'] = tmaze.previous(trials['photostim_on'])
    
    # Stim location
    if 'na' in vr['stim_location'].unique():
        vr['stim_location'] = vr['stim_location'].map(stimid_rename)
    vr['stim_location_num'] = vr['stim_location'].map(stimid_map)
    trials['stim_location_num'] = vr.groupby('trial')['stim_location_num'].max()
    trials['stim_location'] = trials['stim_location_num'].map(stimid_map_inv)
    trials['stim_location_prev'] = tmaze.previous(trials['stim_location'])

# Preprocess
def default_ops():
    ops = {}
    ops['compute_median_trajectory'] = True
    ops['world'] = 'cued_tmaze'
    ops['parent_dir'] = '/n/data2/hms/neurobio/harvey/jonathan/data/behavior'
    ops['virmen_mat_columns'] = ['world_id', 'dx', 'dy', 'dh', 'x', 'y', 'h_int', 'inITI', 'reward', 'dt', 'lick', 'trial']
    # ops['virmen_mat_columns'] = ['world_id', 'dx', 'dy', 'dh', 'x', 'y', 'h_int', 'inITI', 'reward', 'dt', 'lick', 'pitch', 'roll', 'yaw', 'trial']
    ops['is_photostim'] = False
    ops['experiment'] = 'tmaze'
    return ops

def photostim_ops():
    ops = default_ops()
    ops['is_photostim'] = True

def load_vrs(mouse, dates, ops=default_ops()):
    keys = sess.mousedates_to_keys(mouse, dates)
    vrs = [load_vr_behavior_photostim(key, ops) for key in keys]
    return vrs

def define_path(key, ops=default_ops()):
    path = {}
    path['parent_dir'] = ops['parent_dir']
    path['session_dir'] = os.path.join(path['parent_dir'], key['mouse'], key['date'], key['session'])
    path['virmen_mat'] = os.path.join(path['session_dir'], 'virmen', 'sessionData.mat')
    path['vr_preprocessed'] = os.path.join(path['session_dir'], 'vr.pickle')
    if ops['is_photostim']:
        filenames = glob.glob(os.path.join(path['session_dir'], 'photostim', '*.mat'))
        assert len(filenames)==1
        path['photostim_mat'] = filenames[0]
    
    return path

def behavior_stim_data(path):
    mat = scipy.io.loadmat(path['photostim_mat'])
    chNames = ['dt'] + [arr[0] for arr in mat['chNames'][0]]
    df = pd.DataFrame(mat['data'], columns=chNames).iloc[1:]
    df['t'] = df['dt'].cumsum()
    return df

def behavior_stim_location(mat):
    stim_location = pd.Series([trial[1][0][0][0][0] for trial in mat['stimRecord'][0]])
    return stim_location

def import_behavior_stim_data(path):
    mat = scipy.io.loadmat(path['photostim_mat'])
    df = pd.DataFrame()
    df['stim_location'] = behavior_stim_location(mat)
    df['peakLaserVolt'] = mat['peakLaserVolt'][0][0]
    return df

def map_stim_data_to_vr(vr, stim):
    # Get start and stop inds for each stimulation
    start_inds, stop_inds = fc.get_inds_start_stop(vr['photostim_on'], min_len=10, keep_ends=True)
    
    # Check that number is the same in stim data file and vr stim trigger
    lengths_match = (len(start_inds) == len(stop_inds) == len(stim))
    if not lengths_match:
        if (len(stim) == len(start_inds) + 1):# and (stop_inds[-1] == len(vr)-1):
            stim = stim[:-1]
            assert len(start_inds) == len(stop_inds) == len(stim)
        else:
            print('len(start_inds):', len(start_inds))
            print('len(stop_inds):', len(stop_inds))
            print('len(stim):', len(stim))

            twindow = (-15, 15)
            start_inds, stop_inds = fc.get_inds_start_stop(vr['photostim_on'], min_len=10, keep_ends=True)
            tlast = vr['t'][start_inds[-1]]
            plt.figure()
            plt.plot(vr['t'], vr['photostim_on'])
            plt.scatter(vr['t'][start_inds], np.ones(len(start_inds)), c='red')
            plt.xlim(tlast+twindow[0], tlast+twindow[1])
            raise ValueError('VR triggers do not match stim data file.')
    
    # Map stim data to vr time series data
    location = pd.Series('na', index=vr.index)
    volt = pd.Series(0, index=vr.index)
    for i in range(len(stim)):
        slicei = slice(start_inds[i], stop_inds[i])
        location[slicei] = stim.loc[i, 'stim_location']
        volt[slicei] = stim.loc[i, 'peakLaserVolt']
    vr['stim_location'] = location
    vr['laser_volt'] = volt
    return vr

def rename_vr_columns(vr):
    if 'user2' in vr.columns:
        rename_dict = {
            'user0': 'h_pulse',
            'user1': 'h_bias',
            'user2': 'photostim_on'}
    elif 'user0' in vr.columns:
        rename_dict = {
            'user0': 'photostim_on'
        }
    vr = vr.rename(rename_dict, axis=1)
    return vr

def _load_vr_behavior_photostim(key, ops=default_ops()):
    # Load data
    path = define_path(key, ops=ops)
    vr = sess.load_vr(path['virmen_mat'], ops['virmen_mat_columns'])
    
    # Process
    tmaze.preprocess_vr(vr)
    world_dict = tmaze.worlds(ops['world'])
    trials = tmaze.compute_trials_df(vr, world_dict)
    tmaze.compute_vr_maze_variables(vr, trials, world_dict)
    vr.attrs = dict(path=path, ops=ops, trials=trials, **key)

    if ops['is_photostim']:
        # This will check if vr and stim match up
        vr = rename_vr_columns(vr)
        tmaze.add_h_pulse_trial(vr)
        stim = import_behavior_stim_data(path)
        vr = map_stim_data_to_vr(vr, stim)
        vr.attrs['stim'] = stim
    
    if ops['compute_median_trajectory']:
        vr.attrs['traj'] = tmaze.compute_median_trajectory_variables(vr)

    return vr

def load_vr_behavior_photostim(key, ops=default_ops(), recompute=False):
    path = define_path(key)
    if os.path.isfile(path['vr_preprocessed']) and (not recompute):
        vr = fc.load_pickle(path['vr_preprocessed'])
    else:
        vr = _load_vr_behavior_photostim(key, ops=ops)
        fc.save_pickle(vr, path['vr_preprocessed'])

    print("Loaded {mouse} {date} {session}.".format(**key))
    return vr

def main(mouse=None, date=None, session=None, ops=default_ops(), recompute=True):
    key = dict(mouse=mouse, date=date, session=session)
    vr = load_vr_behavior_photostim(key, ops=ops, recompute=recompute)
    return vr

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Load vr session and save pickle.')
    parser.add_argument('--mouse', required=True, type=str)
    parser.add_argument('--date', required=True, type=str)
    parser.add_argument('--session', required=False, type=str, default='session_1')
    parser.add_argument('--ops', required=False, type=str, default='default_ops')
    parser.add_argument('--recompute', required=False, type=bool, default=True)
    
    args = vars(parser.parse_args())
    args['ops'] = locals()[args['ops']]()
    # args['ops'] = getattr(options, args['ops'])()
    main(**args)

