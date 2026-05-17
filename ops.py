def default_ops(imaging=True, env='tmaze', maze='cued_tmaze'):
    ops = {}
    ops['imaging'] = imaging
    ops['env'] = env
    ops['maze'] = maze
    ops['preprocessed_root'] = '/n/data2/hms/neurobio/harvey/jonathan/data/imaging'
    ops['triggers'] = []
    ops['is_photostim'] = False
    if imaging:
        ops['raw_root'] = '/n/scratch3/users/j/jg319/Behavior_Imaging_Data'

        ops['filter1_regex'] = 'filter1'
        ops['filter2_regex']= None
        ops['filter3_regex']= None
        ops['var_filter_key'] = {'G': 'filter1'}

        # X
        ops['activity_signal'] = 'dF/F'
        ops['dilation'] = {'cell_dilation': 0, 'background_dilation': [2, 0]}
        ops['demix'] = None
        ops['min_dist_to_edge'] = 10
        ops['max_val'] = 2**13*0.95
        ops['triggers'].extend(['ScanImage'])

        ops['baseline'] = 'maximin'
        ops['win_baseline'] = 60.0
        ops['sig_baseline'] = 10.0
        ops['prctile_baseline'] = 8.0
        ops['neucoeff'] = 0.7
        ops['tau_s'] = 0.80
        ops['batch_size'] = 500

        ops['fov_microns'] = 680

        ops['varcorr_key'] = None
        
    # vr
    ops['virmen_mat_columns'] = ['world_id', 'dx', 'dy', 'dh', 'x', 'y', 'h_int', 'inITI', 'reward', 'dt', 'lick', 'trial']
    ops['sync_gains'] = {'Ball_pitc': -1,
                'Ball_roll': -1,
                'Ball_yaw': -1,
                'Reward': 1,}

    ops['sync_offsets'] = {'Ball_pitc': 1.4946,
                    'Ball_roll': 1.4965,
                    'Ball_yaw': 1.5,
                  'Reward': 0}

    ops['sync_labels'] = {'Ball_pitc': 'pitch',
                    'Ball_roll': 'roll',
                    'Ball_yaw': 'yaw',
                 'Reward': 'sync_reward'}
    return ops

def ChRmine_GR_ops(**kwargs):
    ops = default_ops(imaging=True, **kwargs)
    ops['filter1_regex'] = 'filter1'
    ops['filter2_regex']= 'filter2*'
    ops['filter3_regex'] = 'filter3*'

    ops['var_filter_key'] = {'G': 'filter1', 'R': 'filter3_reg'}
    return ops

def ChRmine_GR_stim_ops(**kwargs):
    ops = default_ops(imaging=True, **kwargs)
    ops['filter1_regex'] = 'filter1'
    ops['filter2_regex']= 'filter2*'
    ops['filter3_regex'] = 'filter3*'

    ops['var_filter_key'] = {'G': 'filter1', 'R': 'filter2'}
    ops['is_photostim'] = True
    return ops

def ChRmine_BGR_ops(**kwargs):
    ops = default_ops(imaging=True, env='tmaze', maze='cued_tmaze', **kwargs)
    ops['filter1_regex'] = 'filter1'
    ops['filter2_regex']= 'filter2*'
    ops['filter3_regex'] = 'filter3*'
    ops['var_filter_key'] = {'B': 'filter3', 'G': 'filter1', 'R': 'filter2'}

    ops['is_photostim'] = False
    return ops

def ChRmine_BGR_ops2(**kwargs):
    ops = default_ops(imaging=True, env='tmaze', maze='cued_tmaze', **kwargs)
    ops['filter1_regex'] = 'filter1'
    ops['filter2_regex']= 'filter2*'
    ops['filter3_regex'] = 'filter3*'
    ops['var_filter_key'] = {'B': 'filter3', 'G': 'filter2', 'R': 'filter2'}

    ops['is_photostim'] = False
    return ops

def ChRmine_BGR_stim_ops(**kwargs):
    ops = default_ops(imaging=True, env='tmaze', maze='cued_tmaze', **kwargs)
    ops['filter1_regex'] = 'filter1'
    ops['filter2_regex']= 'filter2*'
    ops['filter3_regex'] = 'filter3*'
    ops['var_filter_key'] = {'B': 'filter3', 'G': 'filter1', 'R': 'filter2'}

    ops['is_photostim'] = True
    return ops

def ChRmine_BGR_darkstim_ops(**kwargs):
    ops = default_ops(imaging=True, env=None, maze=None, **kwargs)
    ops['filter1_regex'] = 'filter1'
    ops['filter2_regex']= 'filter2*'
    ops['filter3_regex'] = 'filter3*'
    ops['var_filter_key'] = {'B': 'filter3', 'G': 'filter1', 'R': 'filter2'}

    ops['is_photostim'] = True
    return ops

def Sst44nlsBFP_ops(**kwargs):
    ops = default_ops(**kwargs)
    ops['filter1_regex'] = 'filter1'
    ops['filter2_regex'] = ['filter2_60mW*', 'filter2_25pct*', 'filter2_30pct*', 'filter2_30mW*', 'filter2*']
    ops['var_filter_key'] = {'B': 'filter2_demixed', 'G': 'filter2'}

    ops['is_photostim'] = False
    return ops

def SstCreRFP_Sst44nlsBFP_ops(**kwargs):
    ops = default_ops(**kwargs)
    ops['filter1_regex'] = 'filter1'
    ops['filter2_regex'] = ['filter2_60mW*', 'filter2_30mW*', 'filter2_25pct*', 'filter2_30pct*', 'filter2*']

    ops['var_filter_key'] = {'B': 'filter2_demixed', 'G': 'filter2', 'R': 'filter1'}
    return ops

def Sst44nlsBFP_Lamp5nlsRFP_ops(**kwargs):
    ops = default_ops(**kwargs)
    ops['filter1_regex'] = 'filter1'
    ops['filter2_regex']= ['filter2_60mW*', 'filter2_30mW*', 'filter2*']
    ops['filter3_regex'] = 'filter3*'

    ops['var_filter_key'] = {'B': 'filter3_demixed_reg', 'G': 'filter3_demixed_reg', 'R': 'filter2_reg'}
    # ops['maze'] = 'half_black_half_white'
    return ops

def Lamp5nlsRFP_ops(**kwargs):
    ops = default_ops(**kwargs)
    ops['filter1_regex'] = 'filter1'
    ops['filter2_regex']= 'filter2*'
    ops['filter3_regex'] = None

    ops['var_filter_key'] = {'G': 'filter1', 'R': 'filter2'}
    return ops

def Lamp5nlsRFP_nofilter2_ops(**kwargs):
    ops = default_ops(**kwargs)
    ops['filter1_regex'] = 'filter1'
    ops['filter2_regex']= None
    ops['filter3_regex'] = None

    ops['var_filter_key'] = {'G': 'filter1', 'R': 'filter1_demixed'}
    return ops
