import pandas as pd
import numpy as np
import functions as fc
import scipy.interpolate
import time
import mouse_imaging.behavior as behavior

# ops
def tmaze_opto_ops():
    ops = behavior.default_ops()
    ops['experiment'] = 'tmaze'
    ops['maze'] = 'cued_tmaze'
    ops['virmen_columns_rename'] = {
        'user0': 'opto_out',
        'user1': 'time_out',
    }
    ops['preprocess_fcn'] = 'preprocess_tmaze_opto'
    return ops

def tmaze_heading_pulse_opto_ops():
    ops = default_ops()
    ops['experiment'] = 'tmaze'
    ops['maze'] = 'cued_tmaze'
    ops['virmen_columns_rename'] = {
        'user0': 'opto_out',
        'user1': 'time_out',
        'user2': 'pulse_trial',
        'user3': 'pulse_on',
        'user4': 'pulse_amplitude',
    }
    ops['preprocess_fcn'] = 'preprocess_tmaze_heading_pulse'
    return ops

# preprocess functions
def preprocess_tmaze(vr, ops, preprocess=True):
    world_dict = worlds(ops['maze'])
    vr.attrs['ops']['world_dict'] = world_dict
    
    if preprocess:
        # Only run preprocess once, since this flips h and x.
        preprocess_vr(vr)
    
    vr.attrs['trials'] = compute_trials_df(vr, world_dict)
    compute_vr_maze_variables(vr, vr.attrs['trials'], world_dict)
    
    if ops['do_median_trajectory']:
        traj = compute_median_trajectory_variables(vr)
        vr.attrs['median_trajectory'] = traj

def preprocess_tmaze_opto(vr, ops, preprocess=True):
    preprocess_tmaze(vr, ops, preprocess=preprocess,)
    vr.attrs['trials']['opto_out'] = vr.groupby('trial')['opto_out'].max()
    vr.attrs['trials']['time_out'] = vr.groupby('trial')['time_out'].max()

def preprocess_tmaze_heading_pulse(vr, ops):
    preprocess_tmaze(vr, maze=ops['maze'], preprocess=True, do_median_trajectory=False)
    traj = tmaze.compute_median_trajectory_variables(vr, obs_key={'correct': True, 'pulse_trial': False})
    vr.attrs['median_trajectory'] = traj

    vr.attrs['trials']['opto_out'] = vr.groupby('trial')['opto_out'].max()
    vr.attrs['trials']['time_out'] = vr.groupby('trial')['time_out'].max()

# Define worlds
def worlds(maze):
    if maze is None:
        print('Input cued_tmaze, tower_tmaze, cue_switch.')
        return
    world_dicts = {'cued_tmaze': {
            '1': 'white_right', 
            '2': 'black_left',
            '3': 'white2_right',
            '4': 'black2_left'},
        'tower_tmaze': {
            '1': 'tower_right', 
            '2': 'tower_left'
        },
        'cue_switch': {
            '1': 'white_right', 
            '2': 'black_left',
            '3': 'black_white_right', 
            '4': 'white_black_left'}
    }
    return world_dicts[maze]

# Fix issues in vr data
def extend_trial_into_ITI(trial, inITI, dt, min_len_s=0.4):
    min_len = int(np.ceil(min_len_s / np.nanmean(dt)))
    start_inds, stop_inds = fc.get_inds_start_stop(inITI, min_len=min_len, keep_ends=False)
    inITI_end = stop_inds

    trial_end = np.where(trial.diff() > 0)[0] - 1
    trial_end = trial_end[:len(inITI_end)] # in case recording cuts off before end of last ITI
    
    trial_ = trial.copy()
    for trial_endi, inITI_endi in zip(trial_end, inITI_end):
        trial_[trial_endi:inITI_endi] = trial[trial_endi]
    return trial_

def fix_inITI(vr, max_len_s=1):
    """
    inITI binary signal sometimes fluctuates at the start of the ITI.
    This code ensures there is only one onset and one offset for each trial.
    Does this by removing short (< 1 s) fluctuations down to 0.
    """
    notITI_segments = np.stack(fc.get_inds_start_stop(vr['inITI']==False))
    
    dt = vr['dt'].mean()
    max_len = max_len_s / dt
    notITI_segments_short = notITI_segments[:, np.diff(notITI_segments, axis=0)[0] < max_len]

    # vr['inITI'].iloc[notITI_segments_short[0]] = True
    # Above replaced by below code to avoid SettingWithCopyWarning
    idx = np.zeros(len(vr)).astype(bool)
    idx[notITI_segments_short[0]] = True
    vr.loc[idx, 'inITI'] = True
    
    vr['inITI'] = vr['inITI'].astype(bool)

# Compute individual variables
def _repeat_trial_var(trial_var, trial_len):
    arr = np.concatenate(
        [np.repeat(vari, lengthi) for vari, lengthi in zip(trial_var, trial_len)]
    )
    return arr

def repeat_trial_var(vr, trial_var):
    trial_len = vr.groupby('trial').apply(lambda x: len(x))
    trial_var_timebase = _repeat_trial_var(trial_var, trial_len)
    return trial_var_timebase

def turn(vr):
    vr = vr[vr['inITI'] == 0]
    turn = 'right' if vr.x.iloc[-1] > 0 else 'left'
    return turn

def world(trial, world_dict):
    # trial = trial[trial['inITI'] == 0]
    world = world_dict[str(trial.world_id.iloc[0])]
    return world

def rewarded_side(world):
    side = world.split('_')[-1]
    return side

def correct(trial):
    correct = trial.turn == trial.rewarded_side
    return correct

def trial_type(trial):
    correct_str = {True: 'correct', False: 'incorrect'}
    reward_str = {True: '+reward', False: '-reward'}
    trialtype = f'{trial.world}_{correct_str[trial.correct]}_{reward_str[trial.rewarded_trial]}'
    return trialtype

def target_x(rewarded_side):
    target_x_dict = {'left': -12, 'right': +12}
    target_x = target_x_dict[rewarded_side]
    return target_x

def pos_to_goal_line(obs):
    pos = np.array(obs[['x', 'y']].astype(float))
    target_x = obs['target_x']
    target_line_start = np.array([target_x, 305])
    target_line_end = np.array([target_x, 308])
    nearest, dist = fc.point2line(pos, target_line_start, target_line_end)
    return nearest, dist

def target_y(obs, reward_zone_y_range=(305.5, 307.5)):
    y = float(obs['y'])
    if y > reward_zone_y_range[1]:
        target_y = reward_zone_y_range[1]
    elif y < reward_zone_y_range[0]:
        target_y = reward_zone_y_range[0]
    else:
        target_y = y
    return target_y

def target_dist(obs):
    x_delta = obs['target_x'] - obs['x']
    y_delta = obs['target_y'] - obs['y']
    target_dist = np.hypot(x_delta, y_delta)
    return target_dist

def h_target(obs):
    x_delta = obs['target_x'] - obs['x']
    y_delta = obs['target_y'] - obs['y']
    h_target = fc.wrap_h((np.arctan2(y_delta, x_delta) - np.pi/2)) * -1
    return h_target

def h_target_median_trajectory(obs, bins=15):
    """
    Compute median trajectory value on log1p(target_dist)
    """


    # Split by world
    worlds = obs['world'].unique()
    sr = pd.Series(np.nan, index=obs.index)
    for world in worlds:
        # Bin by log1p(target_dist)
        obsi = obs[obs['world']==world]
        cut = pd.cut(np.log1p(obsi['target_dist']), bins=bins)

        # Compute mean h
        h = obsi.groupby(cut)['h'].apply(fc.circmean)

        # Interpolate to apply to all time points
        
        f = scipy.interpolate.interp1d(dfi.index.mid, dfi['h'], kind='linear', bounds_error=False, fill_value='extrapolate')

        obs_world_idx = obs['world']==world
        sr[obs_world_idx] = f(obs.loc[obs_world_idx, cut_col])
    return sr

def ITI_correct(obs):
    iti_correct = pd.Series('notITI', index=obs.index)
    iti_correct[obs['inITI'] & obs['correct']] = 'ITI_correct'
    iti_correct[obs['inITI'] & ~obs['correct']] = 'ITI_incorrect'
    return iti_correct

def add_h_pulse_trial(vr):
    vr['h_pulse_right'] = vr['h_bias'] > 0
    vr['h_pulse_left'] = vr['h_bias'] < 0
    trial_len = vr.groupby('trial').apply(lambda x: len(x))
    for col in ['h_pulse', 'h_pulse_left', 'h_pulse_right']:
        vr[col + '_trial'] = _repeat_trial_var(vr.groupby('trial')[col].max(), trial_len)

def get_stem_length(vr):
    ymax = vr['y'].max()
    if ymax < 100:
        stem_length = 86
    elif ymax < 200:
        stem_length = 150
    elif ymax > 300:
        stem_length = 300
    return stem_length

def get_reward_zone_y_range(vr):
    stem_length = get_stem_length(vr)
    reward_zone_y_range = (stem_length + 5.5, stem_length + 7.5)
    return reward_zone_y_range

# Median trajectory analysis
def median_trajectory(obs, cols=['x', 'y', 'h'], bins=15, obs_key={'correct': True}, trial_len_quantile=0.25, fix=False):
    cols_noh = [col for col in cols if col!='h']
    # Remove ITI
    obs = obs[~obs['inITI'].astype(bool)]
    if 'h_pulse' in obs.columns:
        print('Omitting h_pulse trials.')
        obs_key.update({'h_pulse_trial': False})
    obs_idx = fc.fetch_index(obs, obs_key)
    obs = obs[obs_idx]
    
    worlds = obs['world'].unique()
    cols = cols + ['world']
    df = pd.DataFrame(columns=cols)
    # Compute trajectory for each world separately
    for world in worlds:
        obsi = obs[obs['world']==world]
        
        # Select shortest quantile of trials
        max_trial_len_s = obsi.groupby('trial')['trial_len_s'].median().quantile(trial_len_quantile)
        obsi = obsi[obsi['trial_len_s'] < max_trial_len_s]

        # Compute median trajectory
        cut = pd.cut(np.log1p(obsi['target_dist']), bins=bins)
        dfi = obsi.groupby(cut)[cols_noh].median()
        dfi['world'] = world
        if 'h' in cols:
            dfi['h'] = obsi.groupby(cut)['h'].apply(fc.circmean)
        df = df.append(dfi)
    if fix:
        df = fix_trajectory(df)
    return df

def fix_trajectory(traj):
    traj2 = pd.DataFrame()
    worlds = traj['world'].unique()
    for world in worlds:
        traji = traj[traj['world']==world].copy()
        turn = world.split('_')[-1]
        if turn == 'right':
            h_wrong = traji.loc[traji['x']>1, 'h'].sort_index(ascending=False) < 0
        elif turn == 'left':
            h_wrong = traji.loc[traji['x']<-1, 'h'].sort_index(ascending=False) > 0
        else:
            print(turn)
        if h_wrong.sum():
            ind0 = h_wrong.index[np.where(h_wrong)[0][0]-1]
            traji.loc[traji.index < ind0, 'h'] = traji.loc[ind0, 'h']
        traj2 = traj2.append(traji)
    return traj2

def interpolate_trajectory(df, bins=309):
    index_new = np.log1p(np.arange(bins))
    worlds = df['world'].unique()
    df_interp = pd.DataFrame()
    for world in worlds:
        dfi = fc.interp_df(df[df['world']==world].drop('world', axis=1), index_new)
        dfi['world'] = world
        df_interp = df_interp.append(dfi)
    return df_interp

def closest_idx(sr, df):
    dfi = df[df['world']==sr['world']]
    if len(dfi) == 0:
        return df.index[0]
    dx = dfi['x'] - sr['x']
    dy = dfi['y'] - sr['y']
    idx = np.hypot(dx, dy).idxmin()
    return idx

def median_trajectory_values(obs, traj, interp_bins=309,):
    # Compute median trajectory, split by world
    traj = interpolate_trajectory(traj, bins=interp_bins)
    traj = traj.dropna(axis=0)
    traj.index = traj.index.astype(str) + '-' + traj['world']
    assert traj.index.is_unique

    # Find closest point on median trajectory
    print('Computing closest index...')
    t0 = time.perf_counter()
    idx = obs.apply(lambda row: closest_idx(row, traj), axis=1)
    t1 = time.perf_counter()
    print('Computed closest index in %.0f s.' %(t1-t0))
    vals = traj.loc[idx]
    vals.index = obs.index

    # Remove datapoints where world is not in trajectory - still some issues it seems 230207
    vals.loc[~obs['world'].isin(traj['world'].unique()).values] = np.nan
    return vals    

def compute_median_trajectory_variables(vr, cols=['x', 'y', 'h', 'dx', 'dy', 'dh'], bins=10, trial_len_quantile=0.25,  obs_key={'correct': True}, fix=False):
    """
    Median trajectory computation.
    """
    print("Computing median trajectory variables.")
    traj = median_trajectory(vr, cols=cols, bins=bins, trial_len_quantile=trial_len_quantile, obs_key=obs_key, fix=fix)
    vals = median_trajectory_values(vr, traj, interp_bins=309,)
    print(vals.columns)
    vals.columns = vals.columns + '_mt'

    for col in vals.columns:
        if col == 'world_mt': continue
        # Add median trajectory value
        vr[col] = vals[col]

        # Add error from median trajectory value
        error = vr[col.rstrip('_mt')] - vr[col]
        if col == 'h_mt':
            error = fc.wrap_h(error)
        vr[col + '_error'] = error
    return traj

# Combine into pipeline
def preprocess_vr(vr):

    # Flip x and h (projected image is flipped horizontally)
    vr['x'] *= -1
    vr['h'] *= -1

    fix_inITI(vr)

    # Trial variable starts from previous trial's ITI
    vr['trial'] = extend_trial_into_ITI(vr['trial'], vr['inITI'], vr['dt'])

def compute_trials_df(vr, world_dict):
    # Compute variables by trial
    trials = pd.DataFrame()

    trials['turn'] = vr.groupby('trial').apply(turn).astype('category')
    trials['world'] = vr.groupby('trial').apply(lambda row: world(row, world_dict)).astype('category')
    
    trials['rewarded_side'] = trials['world'].apply(rewarded_side).astype('category')
    trials['correct'] = trials.apply(correct, axis=1)
    trials['accuracy'] = trials.correct.rolling(window=20).mean()
    trials['target_x'] = trials['rewarded_side'].apply(target_x)
    
    # Compute previous trial variables
    trials['prev_correct'] = behavior.previous(trials['correct'])
    trials['prev_world'] = behavior.previous(trials['world'])

    trials['trial_len_s'] = vr.groupby('trial').apply(lambda x: len(x[x['inITI']==0])) * vr['dt'].mean()

    trials['rewarded_trial'] = vr.groupby('trial').apply(lambda x: x['reward'].max())
    trials['trial_type'] = trials.apply(trial_type, axis=1).astype('category')
    return trials

def map_trial_variables_to_time(vr, trials, world_dict):
    print("Mapping trial variables to time.")
    # Repeat trial variable for length of each trial and add to vr df
    for key in trials.keys():
        if key not in ['trial']:
            trial_len = vr.groupby('trial').apply(lambda x: len(x))
            col = _repeat_trial_var(trials[key], trial_len)
            assert len(col) == len(vr)
            vr[key] = col
            if vr[key].dtype == 'object':
                vr[key] = vr[key].astype('category')

def compute_goal_variables(vr):
    print("Computing goal variables.")
    reward_zone_y_range = get_reward_zone_y_range(vr)
    vr['target_y'] = vr.apply(lambda row: target_y(row, reward_zone_y_range), axis=1)

    # Target distance
    vr['target_dist'] = vr.apply(target_dist, axis=1)
    vr.loc[vr['inITI'].astype(bool), 'target_dist'] = np.nan
    vr['dtarget_dist'] = np.gradient(vr['target_dist'], vr['t'])
    vr['ddtarget_dist'] = np.gradient(vr['dtarget_dist'], vr['t'])

    # Heading error
    vr['h_target'] = h_target(vr)
    # vr.loc[vr['inITI'], 'h_target'] = np.nan
    vr['h_error'] = fc.wrap_h(vr['h'] - vr['h_target'])
    vr['abs_h_error'] = np.abs(vr['h_error'])
    vr['dh_error'] = np.gradient(vr['abs_h_error'], vr['t'])
    vr.loc[np.gradient(vr['trial'])>0, 'dh_error'] = 0 # Remove large change during trial reset
    vr['ddh_error'] = np.gradient(vr['dh_error'], vr['t'])

def compute_movement_variables(vr):
    print("Computing movement variables.")
    # Turning acceleration
    vr['ddh'] = np.gradient(vr['dh'], vr['t'])
    vr['ddh_0.25sigma'] = scipy.ndimage.gaussian_filter1d(vr['ddh'], 0.25/vr['dt'].mean())
    vr['abs_ddh'] = np.abs(vr['ddh'])

def compute_trial_type_variables(vr, world_dict):
    print("Computing trial type variables.")
    # Trial types
    vr['ITI_correct'] = ITI_correct(vr).astype('category')
    if vr['world_id'].dtype == 'int64':
        vr['world'] = vr['world_id'].astype(str).map(world_dict)
    else:
        vr['world'] = vr['world_id'].map(world_dict)
    vr['wall_color'] = vr['world'].str[:5].astype('category')
    vr['trial_type_ITI'] = vr['trial_type'].str[:-8]
    vr.loc[vr['inITI']==1, 'trial_type_ITI'] += '_ITI'

    vr['world'] = vr['world'].astype('category')
    vr['trial_type_ITI'] = vr['trial_type_ITI'].astype('category')

def compute_vr_maze_variables(vr, trials, world_dict):
    map_trial_variables_to_time(vr, trials, world_dict)
    compute_goal_variables(vr)
    compute_movement_variables(vr)
    compute_trial_type_variables(vr, world_dict)

# Run pipeline
def main(self, maze='cued_tmaze', preprocess=True, do_median_trajectory=True):
    world_dict = worlds(maze)
    self.uns['ops']['world_dict'] = world_dict
    
    if preprocess:
        # Only run preprocess once, since this flips h and x.
        preprocess_vr(self.obs)
        preprocess_vr(self.uns['vr'])

    self.uns['trials'] = compute_trials_df(self.uns['vr'], world_dict)

    compute_vr_maze_variables(self.obs, self.uns['trials'], world_dict)
    compute_vr_maze_variables(self.uns['vr'], self.uns['trials'], world_dict)
    
    if do_median_trajectory:
        traj = compute_median_trajectory_variables(self.obs)
        self.uns['median_trajectory'] = traj

