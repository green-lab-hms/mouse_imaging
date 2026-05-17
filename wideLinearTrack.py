import mouse_imaging.behavior as behavior
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ops
def wideLinearTrack_ops():
    ops = behavior.default_ops()
    ops['experiment'] = 'wideLinearTrack'
    ops['world_dict'] = {
            '1': 'neutral'
            }, 
    
    ops['experiment'] = 'wideLinearTrack'
    ops['preprocess_fcn'] = 'preprocess_wideLinearTrack'
    return ops

def wideLinearTrack_photostim_ops():
    ops = wideLinearTrack_ops()
    ops['is_photostim'] = True
    ops['virmen_columns_rename'] = {
            'user0': 'photostim_on',
        }
    return ops

def wideLinearTrack_sidePulse_ops():
    ops = wideLinearTrack_ops()
    ops['virmen_columns_rename'] = {
            'user0': 'pulse_trial',
            'user1': 'is_pulse',
            'user2': 'pulse_bias'
        }
    ops['preprocess_fcn'] = 'preprocess_wideLinearTrack_sidePulse'
    return ops

# Functions
def world(trial, world_dict):
    # trial = trial[trial['inITI'] == 0]
    print(trial['world_id'].iloc[0])
    world = world_dict[str(trial['world_id'].iloc[0])]
    return world

def compute_trials_df(vr,):
    # Compute variables by trial
    trials = pd.DataFrame()
    # trials['world'] = vr.groupby('trial').apply(lambda row: world(row, world_dict)).astype('category')
    trials['trial_len_s'] = vr.groupby('trial').apply(lambda x: len(x[x['inITI']==0])) * vr['dt'].mean()
    trials['rewarded_trial'] = vr.groupby('trial').apply(lambda x: x['reward'].max())
    return trials

# Preprocess functions
def preprocess_wideLinearTrack(vr, ops):
    # invert x
    vr['x'] *= -1
    # vr.attrs['trials'] = compute_trials_df(vr,)

def preprocess_wideLinearTrack_noSavedStimTrigger(vr, ops):
    preprocess_wideLinearTrack(vr, ops)
    vr['photostim_on'] = vr['inITI']
    # vr.attrs['trials'] = compute_trials_df(vr,)

def preprocess_wideLinearTrack_sidePulse(vr, ops):
    preprocess_wideLinearTrack(vr, ops)
    vr['pulse_bias'] *= -1
    vr['pulse_trial'] = vr['pulse_trial'].astype(bool)

# Parsing trials 
def split_session(session_df, chunk_size):
    """
    Split a session into multiple DataFrames of specified chunks of trials size.
    """
    
    trials = session_df.groupby('trial').first().index # all trials
    num_chunks = len(trials) // chunk_size # Round down the division

    chunk_df_arr = []
    for i in range(num_chunks):
        chunk_trial_numbers = trials[i*chunk_size:(i+1)*chunk_size]
        chunk_df = session_df[session_df['trial'].isin(chunk_trial_numbers)]
        chunk_df_arr.append(chunk_df)
            
    return chunk_df_arr

def get_center_trials(session_df, x_range=(-2, 2)):
    """
    Get trials where mouse starts at center of the track.
    """
    trial_first_timestamps = session_df.groupby('trial').first()
    center_start_trial_numbers = trial_first_timestamps[(trial_first_timestamps['x'] >= x_range[0]) & (trial_first_timestamps['x'] <= x_range[1])].index
    return session_df[session_df['trial'].isin(center_start_trial_numbers)]

# Computing trajectories
def select_pulse_trials(session_df, pulse=0,):
    if pulse == 0:
        idy = session_df['pulse_trial'] == False
    else:
        idy = session_df['pulse_trial'] == True
        if pulse == -1:
            idy &= (session_df['pulse_bias'] < 0)
        elif pulse == 1:
            idy &= (session_df['pulse_bias'] > 0)
        else:
            raise ValueError("pulse must be one of [0, -1, 1].")
    pulse_trials = session_df.loc[idy]['trial'].unique()
    session_df_filtered = session_df.loc[session_df['trial'].isin(pulse_trials)].copy()
    return session_df_filtered

def aggregate_by_bin(session_df, cols=['y', 'x'], bin_by='y', bins=np.arange(0, 201, 10), agg_fcn='mean'):
    session_df[f'{bin_by}_bin'] = pd.cut(session_df[bin_by], bins=bins)
    agg_dict = {col: agg_fcn for col in cols}
    trajectory = session_df.groupby(f'{bin_by}_bin').agg(agg_dict)
    return trajectory

# Plotting trajectories
def plot_all_mice_pulse_blocks(mice):
    """
    Plot all left/right pulse blocks of all mice pooled.
    """
    print('All pulse blocks of all mice pooled (center start trials only) with control session')

    right_pulse_arr = []
    right_no_pulse_arr = []
    right_control_arr = []
    right_pulse_bias_arr = []
    
    left_pulse_arr = []
    left_no_pulse_arr = []
    left_control_arr = []
    left_pulse_bias_arr = []
    
    bin_by = 'y'
    bins = np.arange(0, 201, 10)

    for mouse in mice:
        for session in mouse:
            # Remove first trial
            drop_first_trial_df = session[session['trial'] != 1]

            # Split session into 100 trial epochs
            split_df = split_session(drop_first_trial_df, 100)

            # Loop over each 100-trial epoch
            for df in split_df:
                # Pulse direction
                pulse_directions = list(set(np.sign(df['pulse_bias']).unique()) - {0})
                assert len(pulse_directions) == 1
                pulse_direction = pulse_directions[0]

                # First 50 trials are control
                control_block_trial_numbers = df['trial'].unique()[:50]
                control_block_df = df.loc[df['trial'].isin(control_block_trial_numbers)].copy()
                control_block_df = get_center_trials(control_block_df)
            
                # Last 50 trials are pulse
                pulse_block_trial_numbers = df['trial'].unique()[-50:]
                pulse_block_df = df.loc[df['trial'].isin(pulse_block_trial_numbers)].copy()
                pulse_block_df = get_center_trials(pulse_block_df)
                         
                #sorting pulses
                if pulse_direction == 1:
                    right_pulse_trials = select_pulse_trials(session_df, pulse=pulse_direction,)
                    right_pulse
                    right_pulse = get_right_pulse_mean_trajectory(pulse_block_df)
                    right_pulse_arr.append(right_pulse)
                    right_pulse_bias = get_right_pulse_bias(pulse_block_df)
                    right_pulse_bias_arr.append(right_pulse_bias)
                    right_no_pulse = get_no_pulse_mean_trajectory(pulse_block_df)
                    right_control_block = get_mean_trajectory(control_block_df)
                    right_control_arr.append(right_control_block)
                    right_no_pulse_arr.append(right_no_pulse)
                elif pulse_direction == -1:
                    left_pulse = get_left_pulse_mean_trajectory(pulse_block_df)
                    left_pulse_arr.append(left_pulse)
                    left_pulse_bias = get_left_pulse_bias(pulse_block_df)
                    left_pulse_bias_arr.append(left_pulse_bias)
                    left_no_pulse = get_no_pulse_mean_trajectory(pulse_block_df)
                    left_control_block = get_mean_trajectory(control_block_df)
                    left_control_arr.append(left_control_block)
                    left_no_pulse_arr.append(left_no_pulse)

    # Averaging all blocks
    right_pulse['x'] = np.mean([df['x'] for df in right_pulse_arr], axis=0)
    right_pulse['y'] = np.mean([df['y'] for df in right_pulse_arr], axis=0)
    right_pulse_bias['pulse_bias'] = np.mean([df['pulse_bias'] for df in right_pulse_bias_arr], axis=0)
    right_pulse_bias['y'] = np.mean([df['y'] for df in right_pulse_bias_arr], axis=0)
    
    left_pulse['x'] = np.mean([df['x'] for df in left_pulse_arr], axis=0)
    left_pulse['y'] = np.mean([df['y'] for df in left_pulse_arr], axis=0)
    left_pulse_bias['pulse_bias'] = np.mean([df['pulse_bias'] for df in left_pulse_bias_arr], axis=0)
    left_pulse_bias['y'] = np.mean([df['y'] for df in left_pulse_bias_arr], axis=0)
    
    left_no_pulse['x'] = np.mean([df['x'] for df in left_no_pulse_arr], axis=0)
    left_no_pulse['y'] = np.mean([df['y'] for df in left_no_pulse_arr], axis=0)

    right_no_pulse['x'] = np.mean([df['x'] for df in right_no_pulse_arr], axis=0)
    right_no_pulse['y'] = np.mean([df['y'] for df in right_no_pulse_arr], axis=0)
    
    left_control_block['x'] = np.mean([df['x'] for df in left_control_arr], axis=0)
    left_control_block['y'] = np.mean([df['y'] for df in left_control_arr], axis=0)

    right_control_block['x'] = np.mean([df['x'] for df in right_control_arr], axis=0)
    right_control_block['y'] = np.mean([df['y'] for df in right_control_arr], axis=0)

    #plotting
    fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True)
    custom_xlim = (-10, 10)
    custom_ylim = (0, 200)
    plt.setp(axes, xlim=custom_xlim, ylim=custom_ylim)

    axes[0,0].title.set_text('Mean Mouse Trajectory')
    axes[1,0].set(xlabel="x position (vu)", ylabel="y position (vu)")
    axes[0,0].plot(left_control_block['x'], left_control_block['y'], label='Control', color="gray",linestyle='dotted')
    axes[0,0].plot(left_no_pulse['x'], left_no_pulse['y'], label='No Pulse', color="gray")
    axes[0,0].plot(left_pulse['x'], left_pulse['y'], label='Left Pulse', color="blue")
    axes[1,0].plot(right_pulse['x'], right_pulse['y'], label='Right Pulse', color="red")
    axes[1,0].plot(right_no_pulse['x'], right_no_pulse['y'], label='No Pulse', color="gray")
    axes[1,0].plot(right_control_block['x'], right_control_block['y'], label='Control', color="gray",linestyle='dotted')
    axes[0,1].title.set_text('Pulse Bias Strength')
    axes[1,1].set(xlabel="pulse bias")
    axes[1,1].plot(right_pulse_bias['pulse_bias'], right_pulse_bias['y'], label='Right Pulse Bias', color="red")
    axes[0,1].plot(left_pulse_bias['pulse_bias'], left_pulse_bias['y'], label='Left Pulse Bias', color="blue")
    axes[0,0].legend()
    axes[0,1].legend()
    axes[1,0].legend()
    axes[1,1].legend()

    print('number of right pulse blocks: ' + str(len(right_pulse_arr)))
    print('number of left pulse blocks: ' + str(len(left_pulse_arr)))

def plot_average_trajectory_bias(left_pulse_df, left_pulse_bias_df, right_pulse_df, right_pulse_bias_df, no_pulse_df):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
    custom_xlim = (-10, 10)
    custom_ylim = (0, 200)
    plt.setp(axes, xlim=custom_xlim, ylim=custom_ylim)
    
    axes[0].title.set_text('Mean Mouse Trajectory')
    axes[0].set(xlabel="x position (vu)", ylabel="y position (vu)")
    axes[0].plot(left_pulse_df['x'], left_pulse_df['y'], label='Left Pulse', color="blue")
    axes[0].plot(right_pulse_df['x'], right_pulse_df['y'], label='Right Pulse', color="red")
    axes[0].plot(no_pulse_df['x'], no_pulse_df['y'], label='No Pulse', color="gray")
    axes[0].legend()
    axes[1].title.set_text('Pulse Bias Strength')
    axes[1].set(xlabel="pulse bias")
    axes[1].plot(right_pulse_bias_df['pulse_bias'], right_pulse_bias_df['y'], label='Right Pulse Bias', color="red")
    axes[1].plot(left_pulse_bias_df['pulse_bias'], left_pulse_bias_df['y'], label='Left Pulse Bias', color="blue")
    axes[1].legend()
        
    fig.tight_layout()

def plot_mouse_traj_across_sessions(mouse):
    """
    One mouse across individual sessions with control session line plotted (center start trials only)
    """
    center_start_trials_control_df = get_center_trials(mouse[0]).copy()
    control_session = get_mean_trajectory(center_start_trials_control_df)
    
    session_ct = 1
    print('Individual Sessions (center start trials only) with control session')
    
    for session in mouse[1:]: 
        center_start_trials_df = get_center_trials(session)
        
        # Right
        right_pulse = get_right_pulse_mean_trajectory(center_start_trials_df)
        right_pulse_bias = get_right_pulse_bias(center_start_trials_df)
    
        # Left
        left_pulse = get_left_pulse_mean_trajectory(center_start_trials_df)
        left_pulse_bias = get_left_pulse_bias(center_start_trials_df)
    
        # No Pulse
        no_pulse = get_no_pulse_mean_trajectory(center_start_trials_df)
    
        # Plotting
        print("session " + str(session_ct))
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
        custom_xlim = (-10, 10)
        custom_ylim = (0, 200)
        plt.setp(axes, xlim=custom_xlim, ylim=custom_ylim)
        
        axes[0].title.set_text('Mean Mouse Trajectory')
        axes[0].set(xlabel="x position (vu)", ylabel="y position (vu)")
        axes[0].plot(control_session['x'], control_session['y'], label='Control', color="gray",linestyle='dotted')
        axes[0].plot(left_pulse['x'], left_pulse['y'], label='Left Pulse', color="blue")
        axes[0].plot(right_pulse['x'], right_pulse['y'], label='Right Pulse', color="red")
        axes[0].plot(no_pulse['x'], no_pulse['y'], label='No Pulse', color="gray")
        axes[0].legend()
        axes[1].title.set_text('Pulse Bias Strength')
        axes[1].set(xlabel="pulse bias")
        axes[1].plot(right_pulse_bias['pulse_bias'], right_pulse_bias['y'], label='Right Pulse Bias', color="red")
        axes[1].plot(left_pulse_bias['pulse_bias'], left_pulse_bias['y'], label='Left Pulse Bias', color="blue")
        axes[1].legend()
            
        fig.tight_layout()
        session_ct += 1

def plot_trial_start_location(session_df):
    """
    Plot histogram of trial starts.
    """
    trial_first_timestamps = session_df.groupby('trial').first()
    plt.hist(trial_first_timestamps['x'], bins=10)  # Adjust the number of bins as needed
    plt.xlabel('x position (vu)')
    plt.ylabel('Number of Trials')

def plot_mouse_catch_trial_traj_within_sesssion(mouse, session_ind):
    """
    Plot catch trial trajectories in one session with control line.
    """
    center_start_trials_control_df = get_center_trials(mouse[0]).copy()
    control_session = get_mean_trajectory(center_start_trials_control_df)
    plt.plot(control_session['x'], control_session['y'], label='Control', color="gray",linestyle='dotted')
    
    trial_ct = 1
    session = mouse[session_ind]
    
    no_pulse_trials = session.loc[(session['pulse_trial'] == 0)].copy()
    center_start_trials_df = get_center_trials(no_pulse_trials)
    center_start_trial_nums = center_start_trials_df['trial'].unique()
    
    # Get colors from the 'viridis' colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(center_start_trial_nums)))
    color_ct = 0
    
    for trial in center_start_trial_nums:
        no_pulse_trial_df = center_start_trials_df.loc[(center_start_trials_df['trial'] == trial)].copy()
        no_pulse = get_no_pulse_mean_trajectory(no_pulse_trial_df)
        
        plt.xlim(-10, 10)
        plt.ylim(0, 200)
        plt.title('Mean Mouse Trajectory')
        plt.plot(no_pulse_trial_df['x'], no_pulse_trial_df['y'], label='trial '+ str(trial_ct), color=colors[color_ct])
        plt.legend()
        
        trial_ct += 1
        color_ct += 1

def plot_one_mouse_avg_all_sessions_all_trials(mouse):
    """
    Plot trajectory of one mouse, avg all sessions (ALL trials).
    """
    right_pulse_arr = []
    left_pulse_arr = []
    no_pulse_arr = []
    left_pulse_bias_arr = []
    right_pulse_bias_arr = []
    
    for session in mouse[1:]: 
        # Right
        right_pulse = get_right_pulse_mean_trajectory(session)
        right_pulse_arr.append(right_pulse)
        right_pulse_bias = get_right_pulse_bias(session)
        right_pulse_bias_arr.append(right_pulse_bias)
    
        # Left
        left_pulse = get_left_pulse_mean_trajectory(session)
        left_pulse_arr.append(left_pulse)
        left_pulse_bias = get_left_pulse_bias(session)
        left_pulse_bias_arr.append(left_pulse_bias)
    
        # No Pulse
        no_pulse = get_no_pulse_mean_trajectory(session)
        no_pulse_arr.append(no_pulse)
    
    # Averaging all sessions
    right_pulse['x'] = np.mean([df['x'] for df in right_pulse_arr], axis=0)
    right_pulse['y'] = np.mean([df['y'] for df in right_pulse_arr], axis=0)
    right_pulse_bias['pulse_bias'] = np.mean([df['pulse_bias'] for df in right_pulse_bias_arr], axis=0)
    right_pulse_bias['y'] = np.mean([df['y'] for df in right_pulse_bias_arr], axis=0)
    
    left_pulse['x'] = np.mean([df['x'] for df in left_pulse_arr], axis=0)
    left_pulse['y'] = np.mean([df['y'] for df in left_pulse_arr], axis=0)
    left_pulse_bias['pulse_bias'] = np.mean([df['pulse_bias'] for df in left_pulse_bias_arr], axis=0)
    left_pulse_bias['y'] = np.mean([df['y'] for df in left_pulse_bias_arr], axis=0)
    
    no_pulse['x'] = np.mean([df['x'] for df in no_pulse_arr], axis=0)
    no_pulse['y'] = np.mean([df['y'] for df in no_pulse_arr], axis=0)
    
    # Plotting
    print("Average of all sessions (all trials)")
    plot_average_trajectory_bias(left_pulse, left_pulse_bias, right_pulse, right_pulse_bias, no_pulse)

#THIS ONE
def label_pulse_trials(session, block_length=50):
    # First trial does not count
    session.loc[session['trial']==1, 'epoch_pulse_direction'] = 0

    num_trials = session['trial'].max()
    trial_epochs = np.arange(2, num_trials+1, 2+block_length*2)

    session = session[session['trial'] != 1]
    
    # Split session into 100 trial epochs
    split_df = split_session(session, block_length*2)
    
    # Loop over each control-pulse epoch
    for df in split_df:
        # Pulse direction for epoch
        _df = df[df['pulse_bias'].abs()>0.1]
        pulse_directions = np.sign(_df['pulse_bias']).unique()
        if len(pulse_directions) > 1:
            # Sometimes last pulse bleeds through. If so, remove first trial:
            df = df[df['trial'] > df['trial'].iloc[0]]
            _df = df[df['pulse_bias'].abs()>0.1]
            pulse_directions = np.sign(_df['pulse_bias']).unique()
            assert len(pulse_directions) == 1
        pulse_direction_i = pulse_directions[0]
        if pulse_direction_i != pulse_direction:
            continue
    
        # Control or pulse block
        if control_block:
            # First block_length trials are control
            trial_nums = df['trial'].unique()[:block_length]
        else:
            # Last block_length trials are pulse
            trial_nums = df['trial'].unique()[-block_length:]
        block_df = df.loc[df['trial'].isin(trial_nums)].copy()
    
        # Select center trials
        if center_trials:
            block_df = get_center_trials(block_df)
        
        # Select pulse trials
        if control_block or catch_trials:
            block_df = select_pulse_trials(block_df, pulse=0,)
        else:
            block_df = select_pulse_trials(block_df, pulse_direction)
    
        # Aggregate
        dfi = aggregate_by_bin(block_df, cols=['y', metric], bin_by='y', bins=np.arange(0, 201, 10), agg_fcn='mean')
        df_ls.append(dfi)


def mean_pulse_data(mice, metric, pulse_direction, control_block, catch_trials, block_length=50, center_trials=True):
    """
    Plot all left/right pulse blocks of all mice pooled.
    mice: list. List (mice) of list (mouse) of dataframes (session).
    metric: string. e.g. 'x' or 'pulse_bias'
    pulse_direction: int. -1: left pulses, +1: right pulses
    control_block: bool. Control (True) or pulse (False) block.
    catch_trials: bool. Catch trial (no pulse) within pulse block.
    block_length: int. Length of blocks. Assumes control and pulse blocks are the same length.
    center_trials: bool. Whether mouse starts at center or not.
    """
    df_ls = []
    for mouse in mice:
        for session in mouse:
            # Remove first trial
            session = session[session['trial'] != 1]

            # Split session into 100 trial epochs
            split_df = split_session(session, block_length*2)

            # Loop over each control-pulse epoch
            for df in split_df:
                # Pulse direction for epoch
                _df = df[df['pulse_bias'].abs()>0.1]
                pulse_directions = np.sign(_df['pulse_bias']).unique()
                if len(pulse_directions) > 1:
                    # Sometimes last pulse bleeds through. If so, remove first trial:
                    df = df[df['trial'] > df['trial'].iloc[0]]
                    _df = df[df['pulse_bias'].abs()>0.1]
                    pulse_directions = np.sign(_df['pulse_bias']).unique()
                    assert len(pulse_directions) == 1
                pulse_direction_i = pulse_directions[0]
                if pulse_direction_i != pulse_direction:
                    continue

                # Control or pulse block
                if control_block:
                    # First block_length trials are control
                    trial_nums = df['trial'].unique()[:block_length]
                else:
                    # Last block_length trials are pulse
                    trial_nums = df['trial'].unique()[-block_length:]
                block_df = df.loc[df['trial'].isin(trial_nums)].copy()

                # Select center trials
                if center_trials:
                    block_df = get_center_trials(block_df)
                
                # Select pulse trials
                if control_block or catch_trials:
                    block_df = select_pulse_trials(block_df, pulse=0,)
                else:
                    block_df = select_pulse_trials(block_df, pulse_direction)

                # Aggregate
                dfi = aggregate_by_bin(block_df, cols=['y', metric], bin_by='y', bins=np.arange(0, 201, 10), agg_fcn='mean')
                df_ls.append(dfi)

    # Combine
    df_combined = pd.DataFrame()
    df_combined[metric] = np.nanmean([dfi[metric] for dfi in df_ls], axis=0)
    df_combined['y'] = np.nanmean([dfi['y'] for dfi in df_ls], axis=0)
    return df_combined

def plot_all_mice_pulse_blocks2(mice, block_length=50, center_trials=True):
    print('All pulse blocks of all mice pooled (center start trials only) with control session')
    #plotting
    fig, axes = plt.subplots(nrows=2, ncols=2, sharey=True,)
    custom_xlim = (-10, 10)
    custom_ylim = (0, 200)
    plt.setp(axes, xlim=custom_xlim, ylim=custom_ylim)

    # Trajectories
    axes[0,0].title.set_text('Mean Mouse Trajectory')
    axes[1,0].set(xlabel="x position (vu)", ylabel="y position (vu)")
    axes[0,1].title.set_text('Pulse Bias Strength')
    kwargs = dict(block_length=block_length, center_trials=center_trials)

    # Left pulse
    left_control_block = mean_pulse_data(mice, metric='x', pulse_direction=-1, control_block=True, catch_trials=True, **kwargs)
    axes[0,0].plot(left_control_block['x'], left_control_block['y'], label='Control', color='grey',linestyle='dotted')

    left_catch = mean_pulse_data(mice, metric='x', pulse_direction=-1, control_block=False, catch_trials=True, **kwargs)
    axes[0,0].plot(left_catch['x'], left_catch['y'], label='No Pulse', color='grey')

    left_pulse = mean_pulse_data(mice, metric='x', pulse_direction=-1, control_block=False, catch_trials=False, **kwargs)
    axes[0,0].plot(left_pulse['x'], left_pulse['y'], label='Left Pulse', color='tab:blue')

    # Right pulse
    right_control_block = mean_pulse_data(mice, metric='x', pulse_direction=+1, control_block=True, catch_trials=True, **kwargs)
    axes[1,0].plot(right_control_block['x'], right_control_block['y'], label='Control', color='grey',linestyle='dotted')

    right_catch = mean_pulse_data(mice, metric='x', pulse_direction=+1, control_block=False, catch_trials=True, **kwargs)
    axes[1,0].plot(right_catch['x'], right_catch['y'], label='No Pulse', color='grey')

    right_pulse = mean_pulse_data(mice, metric='x', pulse_direction=+1, control_block=False, catch_trials=False, **kwargs)
    axes[1,0].plot(right_pulse['x'], right_pulse['y'], label='Right Pulse', color='tab:red')
    
    # Bias
    axes[1,1].set(xlabel="Pulse bias")

    left_pulse_bias = mean_pulse_data(mice, metric='pulse_bias', pulse_direction=-1, control_block=False, catch_trials=False, **kwargs)
    axes[0,1].plot(left_pulse_bias['pulse_bias'], left_pulse_bias['y'], label='Left Pulse Bias', color='tab:blue')

    right_pulse_bias = mean_pulse_data(mice, metric='pulse_bias', pulse_direction=+1, control_block=False, catch_trials=False, **kwargs)
    axes[1,1].plot(right_pulse_bias['pulse_bias'], right_pulse_bias['y'], label='Right Pulse Bias', color='tab:red')
    
    
    # Legends
    axes[0,0].legend()
    axes[0,1].legend()
    axes[1,0].legend()
    axes[1,1].legend()

    # print('number of right pulse blocks: ' + str(len(right_pulse_arr)))
    # print('number of left pulse blocks: ' + str(len(left_pulse_arr)))


def plot_mouse_pulse_blocks(mouse, session):
    """
    Plot one mouse across pulse blocks within one session with control block line plotted.
    """
    print('Pulse Blocks within one session (center start trials only) with control session')
    
    drop_first_trial_df = mouse[session][mouse[session]['trial'] != 1]
    split_df = split_session(drop_first_trial_df, 100)
    
    for df in split_df:
        center_start_trials_df = get_center_trials(df)
        control_block_trial_numbers = center_start_trials_df['trial'].unique()[:50]
        control_block_df = center_start_trials_df.loc[center_start_trials_df['trial'].isin(control_block_trial_numbers)].copy()
        control_block = get_mean_trajectory(control_block_df)
    
        pulse_block_trial_numbers = center_start_trials_df['trial'].unique()[-50:]
        pulse_block_df = center_start_trials_df.loc[center_start_trials_df['trial'].isin(pulse_block_trial_numbers)].copy()
    
        # Right
        right_pulse = get_right_pulse_mean_trajectory(pulse_block_df)
        right_pulse_bias = get_right_pulse_bias(pulse_block_df)
    
        # Left
        left_pulse = get_left_pulse_mean_trajectory(pulse_block_df)
        left_pulse_bias = get_left_pulse_bias(pulse_block_df)
    
        # No Pulse
        no_pulse = get_no_pulse_mean_trajectory(pulse_block_df)
    
        # Plotting
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
        custom_xlim = (-10, 10)
        custom_ylim = (0, 200)
        plt.setp(axes, xlim=custom_xlim, ylim=custom_ylim)
    
        axes[0].title.set_text('Mean Mouse Trajectory')
        axes[0].set(xlabel="x position (vu)", ylabel="y position (vu)")
        axes[0].plot(control_block['x'], control_block['y'], label='Control', color="gray",linestyle='dotted')
        axes[0].plot(left_pulse['x'], left_pulse['y'], label='Left Pulse', color="blue")
        axes[0].plot(right_pulse['x'], right_pulse['y'], label='Right Pulse', color="red")
        axes[0].plot(no_pulse['x'], no_pulse['y'], label='No Pulse', color="gray")
        axes[0].legend()
        axes[1].title.set_text('Pulse Bias Strength')
        axes[1].set(xlabel="pulse bias")
        axes[1].plot(right_pulse_bias['pulse_bias'], right_pulse_bias['y'], label='Right Pulse Bias', color="red")
        axes[1].plot(left_pulse_bias['pulse_bias'], left_pulse_bias['y'], label='Left Pulse Bias', color="blue")
        axes[1].legend()
            
        fig.tight_layout()

def plot_one_mouse_avg_all_sessions_center_trials(mouse):
    """
    Plot one mouse, avg all sessions (center start trials ONLY).
    """
    right_pulse_arr = []
    left_pulse_arr = []
    no_pulse_arr = []
    left_pulse_bias_arr = []
    right_pulse_bias_arr = []
    
    for session in mouse[1:]: 
        center_start_trials_df = get_center_trials(session)
        
        # Right
        right_pulse = get_right_pulse_mean_trajectory(center_start_trials_df)
        right_pulse_arr.append(right_pulse)
        right_pulse_bias = get_right_pulse_bias(center_start_trials_df)
        right_pulse_bias_arr.append(right_pulse_bias)
    
        # Left
        left_pulse = get_left_pulse_mean_trajectory(center_start_trials_df)
        left_pulse_arr.append(left_pulse)
        left_pulse_bias = get_left_pulse_bias(center_start_trials_df)
        left_pulse_bias_arr.append(left_pulse_bias)
    
        # No Pulse
        no_pulse = get_no_pulse_mean_trajectory(center_start_trials_df)
        no_pulse_arr.append(no_pulse)
    
    # Averaging all sessions
    right_pulse['x'] = np.mean([df['x'] for df in right_pulse_arr], axis=0)
    right_pulse['y'] = np.mean([df['y'] for df in right_pulse_arr], axis=0)
    right_pulse_bias['pulse_bias'] = np.mean([df['pulse_bias'] for df in right_pulse_bias_arr], axis=0)
    right_pulse_bias['y'] = np.mean([df['y'] for df in right_pulse_bias_arr], axis=0)
    
    left_pulse['x'] = np.mean([df['x'] for df in left_pulse_arr], axis=0)
    left_pulse['y'] = np.mean([df['y'] for df in left_pulse_arr], axis=0)
    left_pulse_bias['pulse_bias'] = np.mean([df['pulse_bias'] for df in left_pulse_bias_arr], axis=0)
    left_pulse_bias['y'] = np.mean([df['y'] for df in left_pulse_bias_arr], axis=0)
    
    no_pulse['x'] = np.mean([df['x'] for df in no_pulse_arr], axis=0)
    no_pulse['y'] = np.mean([df['y'] for df in no_pulse_arr], axis=0)
    
    # Plotting
    print("Average of all sessions (center start trials only)")
    plot_average_trajectory_bias(left_pulse, left_pulse_bias, right_pulse, right_pulse_bias, no_pulse)

def plot_one_mouse_across_sessions_no_control(mouse):
    """
    Plot one mouse across individual sessions (center start trials ONLY, NO CONTROL LINE).
    """
    session_ct = 1
    print('Individual Sessions (center start trials only)')
    
    for session in mouse[1:]: 
        center_start_trials_df = get_center_trials(session)
    
        # Right
        right_pulse = get_right_pulse_mean_trajectory(center_start_trials_df)
        right_pulse_bias = get_right_pulse_bias(center_start_trials_df)
    
        # Left
        left_pulse = get_left_pulse_mean_trajectory(center_start_trials_df)
        left_pulse_bias = get_left_pulse_bias(center_start_trials_df)
    
        # No Pulse
        no_pulse = get_no_pulse_mean_trajectory(center_start_trials_df)
    
        # Plotting
        print("session " + str(session_ct))
        plot_average_trajectory_bias(left_pulse, left_pulse_bias, right_pulse, right_pulse_bias, no_pulse)
        session_ct += 1

def plot_all_mice_trajectories(mice):
    """
    Plot to loop through all mice (center start trials ONLY).
    """
    # Create bins for y values
    bins = np.arange(0, 201, 10)
    
    for mouse in mice:
        # Reset the array of session data
        right_pulse_arr = []
        left_pulse_arr = []
        no_pulse_arr = []
        left_pulse_bias_arr = []
        right_pulse_bias_arr = []
        
        for session in mouse: 
            trial_first_timestamps = session.groupby('trial').first()
            center_start_trial_numbers = trial_first_timestamps[(trial_first_timestamps['x'] >= -2) & (trial_first_timestamps['x'] <= 2)].index
            center_start_trials_df = session[session['trial'].isin(center_start_trial_numbers)]
            
            # Right
            right_pulse_trials = center_start_trials_df.loc[(center_start_trials_df['pulse_trial'] == 1) & (center_start_trials_df['pulse_bias'] > 0)]['trial'].unique()
            filtered_right_session = center_start_trials_df.loc[center_start_trials_df['trial'].isin(right_pulse_trials)].copy()
            filtered_right_session['y_bin'] = pd.cut(filtered_right_session['y'], bins=bins)
            right_pulse = filtered_right_session.groupby('y_bin').agg({'y': 'mean', 'x': 'mean'})
            right_pulse_arr.append(right_pulse)
    
            right_pulse_bias = filtered_right_session.groupby('y_bin').agg({'y': 'mean', 'pulse_bias': 'mean'})
            right_pulse_bias_arr.append(right_pulse_bias)
        
            # Left
            left_pulse_trials = center_start_trials_df.loc[(center_start_trials_df['pulse_trial'] == 1) & (center_start_trials_df['pulse_bias'] < 0)]['trial'].unique()
            filtered_left_session = center_start_trials_df.loc[center_start_trials_df['trial'].isin(left_pulse_trials)].copy()
            filtered_left_session['y_bin'] = pd.cut(filtered_left_session['y'], bins=bins)
            left_pulse = filtered_left_session.groupby('y_bin').agg({'y': 'mean', 'x': 'mean'})
            left_pulse_arr.append(left_pulse)

            left_pulse_bias = filtered_left_session.groupby('y_bin').agg({'y': 'mean', 'pulse_bias': 'mean'})
            left_pulse_bias_arr.append(left_pulse_bias)
        
            # No Pulse
            no_pulse_trials = center_start_trials_df.loc[(center_start_trials_df['pulse_trial'] == 0)].copy()
            no_pulse_trials['y_bin'] = pd.cut(no_pulse_trials['y'], bins=bins)
            no_pulse = no_pulse_trials.groupby('y_bin').agg({'y': 'mean', 'x': 'mean'})
            no_pulse_arr.append(no_pulse)
        
        # Averaging all sessions
        right_pulse['x'] = np.mean([df['x'] for df in right_pulse_arr], axis=0)
        right_pulse['y'] = np.mean([df['y'] for df in right_pulse_arr], axis=0)
        
        left_pulse['x'] = np.mean([df['x'] for df in left_pulse_arr], axis=0)
        left_pulse['y'] = np.mean([df['y'] for df in left_pulse_arr], axis=0)
        
        no_pulse['x'] = np.mean([df['x'] for df in no_pulse_arr], axis=0)
        no_pulse['y'] = np.mean([df['y'] for df in no_pulse_arr], axis=0)
        
        # Plotting
        variable_name = get_variable_name(mouse)
        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True)
        custom_xlim = (-10, 10)
        custom_ylim = (0, 200)
        plt.setp(axes, xlim=custom_xlim, ylim=custom_ylim)
    
        print("Avg all sessions for " + variable_name)
        axes[0].title.set_text('Mean Mouse Trajectory')
        axes[0].set(xlabel="x position (vu)", ylabel="y position (vu)")
        axes[0].plot(left_pulse['x'], left_pulse['y'], label='Left Pulse', color="blue")
        axes[0].plot(right_pulse['x'], right_pulse['y'], label='Right Pulse', color="red")
        axes[0].plot(no_pulse['x'], no_pulse['y'], label='No Pulse', color="gray")
        axes[0].legend()
        axes[1].title.set_text('Pulse Bias Strength')
        axes[1].set(xlabel="pulse bias")
        axes[1].plot(right_pulse_bias['pulse_bias'], right_pulse_bias['y'], label='Right Pulse Bias', color="red")
        axes[1].plot(left_pulse_bias['pulse_bias'], left_pulse_bias['y'], label='Left Pulse Bias', color="blue")
        axes[1].legend()
        
        fig.tight_layout()
    