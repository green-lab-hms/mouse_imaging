import glob, os, importlib
from mouse_imaging import functions as fc
from mouse_imaging import sess, tmaze
import numpy as np
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

# Options
def default_ops():
    ops = {}
    ops['parent_dir'] = '/n/data2/hms/neurobio/harvey/jonathan/data/behavior'
    ops['virmen_mat_columns'] = ['world_id', 'dx', 'dy', 'dh', 'x', 'y', 'h_int', 'inITI', 'reward', 'dt', 'lick', 'trial']
    # ops['virmen_mat_columns'] = ['world_id', 'dx', 'dy', 'dh', 'x', 'y', 'h_int', 'inITI', 'reward', 'dt', 'lick', 'pitch', 'roll', 'yaw', 'trial']
    ops['is_photostim'] = False
    ops['virmen_columns_rename'] = {}
    
    ops['do_median_trajectory'] = False
    ops['preprocess_fcn'] = 'preprocess_data'
    return ops

# Loading functions
def save_vr(vr):
    fc.save_pickle(vr, vr.attrs['path']['vr_preprocessed'])

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

def load_vr(key, ops, recompute=False, save=True):
    path = define_path(key, ops)
    if os.path.isfile(path['vr_preprocessed']) and  not recompute:
        vr = fc.load_pickle(path['vr_preprocessed'])
    else:
        print("Processing {mouse} {date} {session}".format(**key))
        vr = sess.load_vr(path['virmen_mat'], columns=ops['virmen_mat_columns'])
        vr.rename(ops['virmen_columns_rename'], inplace=True, axis=1)
        # vr['x'] *= -1
        vr.attrs['key'] = key
        vr.attrs['ops'] = ops
        vr.attrs['path'] = path
        experiment_module = importlib.import_module(f"mouse_imaging.{ops['experiment']}")
        preprocess_fcn = getattr(experiment_module, ops['preprocess_fcn'])
        preprocess_fcn(vr, ops)
            
        if ops['is_photostim']:
            stim = import_behavior_stim_data(path)
            vr = map_stim_data_to_vr(vr, stim)
            vr.attrs['stim'] = stim

        if save: save_vr(vr)
    print("Loaded {mouse} {date} {session}".format(**key))
    return vr

def load_vrs(mouse, dates, ops=default_ops(), recompute=False):
    keys = sess.mousedates_to_keys(mouse, dates)
    vrs = [load_vr(key, ops, recompute=recompute) for key in keys]
    return vrs
    
# Photostim functions
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
    df.index += 1 # 0-based indexing
    return df

def map_stim_data_to_vr(vr, stim):
    # Get start and stop inds for each stimulation
    start_inds, stop_inds = fc.get_inds_start_stop(vr['photostim_on'], min_len=0, keep_ends=True)
    
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
            start_inds, stop_inds = fc.get_inds_start_stop(vr['photostim_on'], min_len=0, keep_ends=True)
            tlast = vr['t'][start_inds[-1]]
            plt.figure()
            plt.plot(vr['t'], vr['photostim_on'])
            plt.scatter(vr['t'][start_inds], np.ones(len(start_inds)), c='red')
            plt.xlim(tlast+twindow[0], tlast+twindow[1])
            raise ValueError('VR triggers do not match stim data file.')
    
    # Map stim data to vr time series data
    location = pd.Series('na', index=vr.index)
    volt = pd.Series(0, index=vr.index, dtype=np.float64)
    for i in range(len(stim)):
        slicei = slice(start_inds[i], stop_inds[i])
        location[slicei] = stim.loc[i+1, 'stim_location'] # 1-based indexing
        volt[slicei] = stim.loc[i+1, 'peakLaserVolt'] # 1-based indexing
    vr['stim_location'] = location
    vr['laser_volt'] = volt
    return vr

# Trial functions
def _repeat_trial_var(trial_var, trial_len):
    arr = np.concatenate(
        [np.repeat(vari, lengthi) for vari, lengthi in zip(trial_var, trial_len)]
    )
    return arr

def repeat_trial_var(vr, trial_var):
    trial_len = vr.groupby('trial').apply(lambda x: len(x))
    trial_var_timebase = _repeat_trial_var(trial_var, trial_len)
    return trial_var_timebase

def previous(sr):
    prev = np.concatenate([pd.Series('none'), sr.values[:-1].astype(str)])
    return prev

# Analysis
def h_target(x, y, x_target, y_target):
    x_delta = x_target - x
    y_delta = y_target - y
    h_target = fc.wrap_h((np.arctan2(y_delta, x_delta) - np.pi/2)) * -1
    return h_target

# Command-line interface
def main(mouse, date, session='session_1', ops=default_ops(), save=True):
    key = dict(mouse=mouse, date=date, session=session)
    vr = load_vr(key, ops, recompute=True, save=save)
    return vr

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Load session as anndata and save pickle.')
    parser.add_argument('--mouse', required=True, type=str)
    parser.add_argument('--date', required=True, type=str)
    parser.add_argument('--session', required=False, type=str, default='session_1')
    parser.add_argument('--ops', required=False, type=str, default='default_ops')
    args = vars(parser.parse_args())
    args['ops'] = locals()[args['ops']]()
    main(**args)
