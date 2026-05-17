import os, warnings
import numpy as np
import pandas as pd
import functions as fc
import mouse_imaging.analysis as an
import mouse_imaging.session as sess
import scipy.stats
import scipy.io
import importlib
importlib.reload(an)
# Handy functions
def unpack_stim_sequence_targets(adata):
    sequence = adata.uns['photostim_sequence']
    targets = adata.uns['photostim_targets']
    return sequence, targets

# Metadata
def get_photostim_sequence(metadata):
    photostim_sequence = np.array(list(map(int, metadata['SI.hPhotostim.sequenceSelectedStimuli'].strip("[]").split())))-1
    return photostim_sequence

def stim_name_to_group(name):
    group = int(name.split('Group')[1].split(' ')[1])
    return group

def t_to_obs_col(obs, col, t, offset_s=0):
    idy = np.where(obs['t']>=(t+offset_s))[0][0]
    val = obs[col].iloc[idy]
    return val

def add_photostim_sequence_cols(adata, obs_cols):
    sequence, targets = unpack_stim_sequence_targets(adata)
    for col in obs_cols:
        sequence[col] = sequence['t'].apply(lambda t: t_to_obs_col(adata.obs, col, t))

def stim_name_to_group_dict(md):
    names = [roi['name'] for roi in md['RoiGroups']['photostimRoiGroups'][1:]]
    group_dict = {iroi+1: stim_name_to_group(names[iroi]) for iroi in range(len(names))}
    return group_dict

def galvo_to_pixel(galvo, md):
    Ly_pixel = int(md['SI.hRoiManager.linesPerFrame'])
    Lx_pixel = int(md['SI.hRoiManager.pixelsPerLine'])
    Ly_galvo, Lx_galvo = md['RoiGroups']['imagingRoiGroup']['rois']['scanfields']['sizeXY']
    pixel = galvo.copy()
    pixel[:, 0] = ( galvo[:, 0] * Lx_pixel / Lx_galvo + Lx_pixel / 2 ).astype(int)
    pixel[:, 1] = ( galvo[:, 1] * Ly_pixel / Ly_galvo + Ly_pixel / 2 ).astype(int)
    return pixel

def roi_to_leftout_source_dict(targets, groups=None):
    if groups is None:
        groups = targets['group'].unique()
    d = {}
    for group in groups:
        rois = targets[targets['group']==group]['roi'].unique()
        sources = targets[targets['group']==group]['nearest_source'].unique()
        for roi in rois:
            roi_sources = targets[targets['roi']==roi]['nearest_source'].unique()
            leftout_source = set(sources) - set(roi_sources)
            assert len(leftout_source) == 1
            d[roi] = list(leftout_source)[0]
    return d

def photostim_targets(self, maxdist=4, plane=1):
    assert self.uns['metadata']['nslices'] == 1 # More than one plane not yet supported

    # First RoiGroup is a dummy, so start from second
    df = pd.DataFrame()
    for iroi, roi in enumerate(self.uns['metadata']['RoiGroups']['photostimRoiGroups'][1:]):
        if self.uns['ops']['slm_photostim']:
            galvo = np.array(roi['rois'][0]['scanfields']['slmPattern'])
            pixel = galvo_to_pixel(galvo, self.uns['metadata'])
            dfi = pd.DataFrame(pixel, columns=['x', 'y', 'z', 'power'])
            dfi['group'] = stim_name_to_group(roi['name'])
        else:
            galvo = np.array(roi['rois'][1]['scanfields']['centerXY'])[np.newaxis, :]
            pixel = galvo_to_pixel(galvo, self.uns['metadata'])
            dfi = pd.DataFrame(pixel, columns=['x', 'y'])
        dfi['roi'] = iroi + 1
        dfi['roi_name'] = roi['name']
        
        df = df.append(dfi)

    source_meds = np.array([stati['med'] for stati in self.stat])
    yx = df[['y', 'x']].to_numpy()
    dists = np.array([np.hypot(*(source_meds - yxi).T) for yxi in yx])
    df['min_dist'] = dists.min(axis=1)
    df['nearest_source'] = [f'plane{plane}_source{isource}' for isource in np.argmin(dists, axis=1)]
    # if not (df.loc[df['group'] < df['group'].max(), 'min_dist'] < maxdist).all():
    #     warnings.warn(f'Some photostimulation targets are greater than {maxdist} pixels from closest source.')
    return df

def photostim_sequence(self, min_dist_s=1):
    df = pd.DataFrame()
    dt = 1/self.uns['metadata']['volume_rate']
    min_dist = int(min_dist_s / dt)
    stim_inds = fc.rising_idx(self.sync.raw['stimTrigg'], min_dist=min_dist)
    assert len(stim_inds) > 0
    df['t'] = self.sync.raw['t'].iloc[stim_inds].to_numpy()
    sequence = get_photostim_sequence(self.uns['metadata'])
    if len(sequence) >= len(df):
        df['roi'] = sequence[:len(df)]
        if self.uns['ops']['slm_photostim']:
            group_dict = stim_name_to_group_dict(self.uns['metadata'])
            df['group'] = df['roi'].map(group_dict)
    else:
        warnings.warn('Sequence of photostimulation groups is shorter than the number of photostimulation triggers.')
    return df

def add_to_var(self, targets, microns_per_pixel=None):
    if self.uns['ops']['slm_photostim']:
        # Assign stim group to sources
        for group in targets['group'].unique():
            group_id = 'stim_group%i' %group
            self.var[group_id] = False
            cells = targets.loc[targets['group']==group, 'nearest_source']
            self.var.loc[cells, group_id] = True
    
    # Assign target rois for each cell depending on distance threshold
    # self.var['target_rois'] = cells_to_target_rois(self, min_dist_um=0)
    # self.var['target_rois_>%ium' %min_dist_um] = cells_to_target_rois(self, min_dist_um=min_dist_um)
    # self.var['min_dist_to_target_um'] = min_dist_to_photostim_target(self.var, targets, photostim_group=1, microns_per_pixel=microns_per_pixel)

def process_photostim_session(self):
    # Get pixel to micron conversion
    fov_microns = self.uns['ops']['fov_microns']
    fov_pixels = self.uns['metadata']['Lx']
    self.uns['metadata']['microns_per_pixel'] = microns_per_pixel = fov_microns / fov_pixels

    # Compute target and sequence dataframes 
    self.uns['photostim_targets'] = targets = photostim_targets(self, maxdist=4)
    self.uns['photostim_sequence'] = photostim_sequence(self,)

    # Add data to var
    add_to_var(self, targets, microns_per_pixel=microns_per_pixel)

# Calculating distances to targets

def dist_to_points(x, y, x_points, y_points):
    dx = x_points - x
    dy = y_points - y
    dist = np.hypot(dx, dy)
    return dist

# def dist_to_closest_point(x, y, x_points, y_points):
#     dist = dist_to_points(x, y, x_points, y_points)
#     min_dist = dist.min()
#     return min_dist

def target_nearest_sources_is_stimulated(adata, max_pval=0.01):
    nearest_sources = adata.uns['photostim_targets']['nearest_source'].unique()
    sources_stimulated = pd.Series(False, index=nearest_sources)
    sequence = adata.uns['photostim_sequence']
    for var_name in nearest_sources:
        if var_name in adata.var_names:
            dX = compute_dX(adata[:, var_name], tlim0=(-1, 0), tlim=(0, 1), min_dist_um=0,)
            group1 = dX[0, sequence['group']==1]
            group2 = dX[0, sequence['group']==2]
            
            if scipy.stats.ranksums(group1, group2).pvalue < max_pval:
                sources_stimulated[var_name] = True
    return sources_stimulated

def dist_to_targets_1cell(adata, var_name, group=1, min_dist_um=50, targets_significant_stim=None):
    targets = adata.uns['photostim_targets']

    # Extract rois that are >min_dist_um from cell
    cell_rois = cell_to_target_rois(var_name, adata, min_dist_um=min_dist_um)
    
    # Choose target entries that are in allowable cell_rois and in group
    idy = targets['roi'].isin(cell_rois) & (targets['group'] == group)
    if targets_significant_stim is not None:
        idy = idy & targets_significant_stim

    # Extract x, y targets that meet criteria
    x_targets, y_targets = targets.loc[idy, ['x', 'y']].values.T
    
    # If no stim targets > min_dist_um, return nan
    if len(x_targets) == 0:
        return np.array([np.nan])
    
    # Compute distance btw cell and stim targets
    x, y = adata.var.loc[var_name, ['x', 'y']].values.T
    dist = dist_to_points(x, y, x_targets, y_targets)
    dist_um = dist * adata.uns['metadata']['microns_per_pixel']
    
    if (dist_um < min_dist_um).any():
        print(dist_um)
    return dist_um

def dist_to_closest_target(adata, group=1, min_dist_um=40, require_significant_stim=False):
    if require_significant_stim:
        nearest_sources_stim = target_nearest_sources_is_stimulated(adata, max_pval=0.01)
        targets_significant_stim = adata.uns['photostim_targets']['nearest_source'].map(nearest_sources_stim)
    else:
        targets_significant_stim = None
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        dist = [np.nanmin(dist_to_targets_1cell(adata, var_name, group=group, min_dist_um=min_dist_um, targets_significant_stim=targets_significant_stim)) for var_name in adata.var_names]
    dist = pd.Series(dist, index=adata.var_names)
    return dist

def dist_to_closest_photostim_target_deprecated(adata, stim_group=1, min_dist_um=30):
    targets = adata.uns['photostim_targets']
    x_targets, y_targets = targets.loc[targets['group']==stim_group, ['x', 'y']].values.T
    x, y = adata.var[['x', 'y']].values.T
    min_dist_thresh = min_dist_um / adata.uns['metadata']['microns_per_pixel']
    dist_to_closest_target = np.array([dist_to_closest_point(xi, yi, x_targets, y_targets, min_dist=min_dist_thresh) for xi, yi in zip(x, y)])
    min_dist_um = dist_to_closest_target * adata.uns['metadata']['microns_per_pixel']
    sr = pd.Series(min_dist_um, index=adata.var_names)
    return sr

def min_dist_to_other_stim_targets_deprecated(stim_targets, adata):
    assert len(stim_targets['group'].unique()) == 1
    target_sources = stim_targets['nearest_source'].unique()
    sr = pd.Series(index=target_sources, dtype=float)
    for target_source in target_sources:
        x, y = adata.var.loc[target_source, ['x', 'y']]
        x_other, y_other = stim_targets.loc[stim_targets['leftout_source']==target_source, ['x', 'y']].values.T
        sr[target_source] = dist_to_closest_point(x, y, x_other, y_other) * adata.metadata['microns_per_pixel']
    return sr

def targets_center_of_mass(adata, stim_group=1):
    """
    Note: Each target is repeated ntargets-1 times because of the rotating leave-one-out design.
    However, we can still take the mean across all targets since they are represented equally.
    """
    targets = adata.uns['photostim_targets']
    targets = targets[targets['group']==stim_group]
    x_center, y_center = targets['x'].mean(), targets['y'].mean()
    return x_center, y_center

def dist_to_targets_center_of_mass(adata, stim_group=1):
    x_center, y_center = targets_center_of_mass(adata, stim_group=1)
    x, y = adata.var[['x', 'y']].values.T
    dx = x - x_center
    dy = y - y_center
    dist = pd.Series(np.hypot(dx, dy), index=adata.var_names)
    return dist

# Triggering
def compute_photostim_Xtrig(adata, t_range=(-5, 10)):
    targets = adata.uns['photostim_targets']
    sequence = adata.uns['photostim_sequence']
    
    trig = an.trigger_inds(adata.obs, trigger_t=sequence['t'], t_range=t_range, safe=False)
    Xtrig = an.fetch_X(adata.X, trig['idyx'])
    return Xtrig, trig

def index_Xtrig(Xtrig, sources=None, stim_group=None, adata=None, leftout=True):
    sequence, targets = unpack_stim_data(adata)
    ls = []
    for source in sources:
        idx1 = sequence['group'] == stim_group
        if leftout is not None:
            if leftout:
                idx1 &= (sequence['leftout_source'] == source)
            else:
                idx1 &= (sequence['leftout_source'] != source)
        ls.append(Xtrig[adata.var_names==source, idx1])
    idx1_len = max([l.shape[1] for l in ls])
    Xtrig_sel = np.zeros((len(sources), idx1_len, Xtrig.shape[2]))
    Xtrig_sel[:] = np.nan
    for i, arr in enumerate(ls):
        Xtrig_sel[i, :len(arr)] = arr
    return Xtrig_sel

def photostim_idx(obs, adata, after_s=1, before_s=0, stim_group=1):
    sequence = adata.uns['photostim_sequence']
    t = sequence['t'][sequence['group']==stim_group]
    idx = np.zeros(len(obs)).astype(bool)
    after_inds = int(after_s / obs['dt'].mean())
    before_inds = int(before_s / obs['dt'].mean())
    for t0 in t:
        ind0 = np.where(obs['t']>=t0)[0][0]
        idx[ind0-before_inds:ind0+after_inds] = True
    return idx

# Influence (deprecated)
def influence_value(X, group, norm=False):
    group = np.array(group)
    diff = X[:, group==1].mean(axis=1) - X[:, group==2].mean(axis=1)
    if norm:
        diff = diff / X[:, group==2].std(axis=1)
    return diff

def shuffle_groups(X, group, metric, n_shuffles=10000):
    group = np.array(group)
    shuffle = np.zeros((len(X), n_shuffles))
    shuffle[:] = np.nan
    for i in range(n_shuffles):
        group_shuffled = group.copy()
        random.shuffle(group_shuffled)
        shuffle[:, i] = metric(X, group_shuffled)
    return shuffle
    
def influence_pvalue(X, group, n_shuffles=10000):
    diff = influence_value(X, group).reshape((len(X), 1))
    diff_shuffle = shuffle_groups(X, group, influence_value)
    
    percentile = np.array([scipy.stats.percentileofscore(diff_shuffle[i], diff[i]) for i in range(len(diff))]) / 100
    pvalue = np.stack([percentile, 1 - percentile]).min(axis=0) * 2

    # if lower than all shuffles, set p-value to 1/n_shuffles
    min_pvalue = 1 / n_shuffles
    pvalue[pvalue==0] = min_pvalue
    return pvalue

def compute_influence(adata, max_trials=100):
    sequence = adata.uns['photostim_sequence'].iloc[:max_trials]
    X = an.X_time_diff(adata, sequence['t'])
    adata.var['influence_pvalue'] = an.influence_pvalue(X, photostim_sequence['group'])
    adata.var['influence_-log10pvalue'] = -np.log10(adata.var['influence_pvalue'])
    adata.var['influence_value'] = an.influence_value(X, photostim_sequence['group'], norm=True)

# Influence measurement
def cell_to_target_rois(cell, adata, min_dist_um=30):
    sequence, targets = unpack_stim_sequence_targets(adata)
    rois = targets['roi'].unique()
    rois_pass = []
    for roi in rois:
        targets_roi = targets[targets['roi']==roi]
        if min_dist_um is None:
            rois_pass.append(roi)
        else:
            min_dist_to_targets = dist_to_points(adata.var.loc[cell, 'x'], adata.var.loc[cell, 'y'], targets_roi['x'], targets_roi['y']).min()
            min_dist_to_targets_um = min_dist_to_targets * adata.uns['metadata']['microns_per_pixel']
            if min_dist_to_targets_um > min_dist_um:
                rois_pass.append(roi)
    return rois_pass

def cells_to_target_rois(adata, min_dist_um=30):
    rois = pd.Series([cell_to_target_rois(cell, adata, min_dist_um=min_dist_um) for cell in adata.var.index], index=adata.var.index)
    return rois

def roi_sequence(sequence, rois):
    seq = np.zeros(len(sequence)).astype(bool)
    for roi in rois:
        seq[sequence['roi']==roi] = True
    return seq

def compute_dX(adata, tlim0=(-1, 0), tlim=(0, 1), min_dist_um=40, layer='dcnv_norm'):
    """
    Computes change in activity relative to stimulation onset (Mean of tlim - tlim0).
    Additionally removes cell, trial entries where the cell is less than min_dist_um from closest photostimulation target.

    adata: anndata imaging session object.
    tlim0: time range to compute baseline.
    tlim: time range to compute stimulation effect.
    min_dist_um: minimum distance that a cell can be from a photostimulation target.

    returns: dX, with shape: (cells, trials)
    """
    # Compute change in activity
    sequence, targets = unpack_stim_sequence_targets(adata)
    dX = an.X_time_diff(adata, sequence['t'], layer=layer, tlim=tlim, tlim0=tlim0) # shape: (cells, trials)

    # Omit cell, trial pairs where cell is too close to target
    if min_dist_um is not None:
        target_rois = cells_to_target_rois(adata, min_dist_um=min_dist_um) # shape: (cells,), each entry is a list of acceptable rois
        keep_idx = np.stack([roi_sequence(sequence, rois) for rois in target_rois]) # shape: (cells, trials), each row is a True/False array of acceptable trials
        # print(f'Omitting %i cell, trial pairs.' %(keep_idx==False).sum())
        dX[~keep_idx] = np.nan

    return dX

def compute_influence_value(adata, obs_key=None, stim_group=1, ctl_group=2, max_trials=200, zscore=False, **kwargs):
    """
    Computes influence value as ( mean(stim trials) - mean(ctl trials) ) / std(ctl trials).
    Stim and control trials are both selected for a given trial variable before this computation. 
    """
    adata = adata.copy()
    adata.uns['photostim_sequence'] = adata.uns['photostim_sequence'].iloc[:max_trials]
    dX = compute_dX(adata, **kwargs)
    stim_idx = (adata.uns['photostim_sequence']['group']==stim_group)
    ctl_idx = (adata.uns['photostim_sequence']['group']==ctl_group)
    if obs_key is not None:
        add_photostim_sequence_cols(adata, obs_key.keys())
        obs_idx = an.fetch_index(adata.uns['photostim_sequence'], obs_key)
        stim_idx &= obs_idx
        ctl_idx &= obs_idx
    
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        influence_pertrial = (dX[:, stim_idx] - np.nanmean(dX[:, ctl_idx], axis=1)[:, None]) 

        std = np.nanstd(dX[:, ctl_idx], axis=1)
        std[std == 0] = np.nan
        if zscore:
            print('Z-scoring influence value.')
            influence_pertrial /= std[:, None]
        else:
            print('Not Z-scoring influence value.')
            influence_pertrial[std==0, :] = np.nan
        influence = np.nanmean(influence_pertrial, axis=1)

    pvalue = []
    for influence_celli in influence_pertrial:
        if (influence_celli == 0).all():
            pvaluei = np.nan
        else:
            pvaluei = scipy.stats.wilcoxon(influence_celli).pvalue
        pvalue.append(pvaluei)
    pvalue = np.array(pvalue)
    
    ntrials_stim = (~np.isnan(dX[:, stim_idx])).sum(axis=1)
    ntrials_ctl = (~np.isnan(dX[:, ctl_idx])).sum(axis=1)

    return influence, pvalue, ntrials_stim, ntrials_ctl # shape: (cells,)

def compute_activity_value(adata, obs_key=None, ctl_group=2, max_trials=200, **kwargs):
    """
    Computes activity value as mean(ctl trials).
    """

    adata = adata.copy()
    adata.uns['photostim_sequence'] = adata.uns['photostim_sequence'].iloc[:max_trials]
    dX = compute_dX(adata, **kwargs)
    ctl_idx = (adata.uns['photostim_sequence']['group']==ctl_group)
    if obs_key is not None:
        add_photostim_sequence_cols(adata, obs_key.keys())
        obs_idx = an.fetch_index(adata.uns['photostim_sequence'], obs_key)
        stim_idx &= obs_idx
        ctl_idx &= obs_idx
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        activity_value = np.nanmean(dX[:, ctl_idx], axis=1)
    return activity_value

# Photostim-average frames
def join_df_by_t(df, obs):
    def get_obs_row(obs, ti):
        ind = np.where(obs['t_obs'] >= ti)[0][0]
        return obs.iloc[ind]
    
    df2 = pd.DataFrame()
    obs = obs.rename({'t': 't_obs'}, axis=1)
    for _, df_row in df.iterrows():
        obs_row = get_obs_row(obs, df_row['t'])
        row = df_row.append(obs_row)
        df2 = df2.append(row, ignore_index=True)
    return df2

def export_photostim_average_frames_deprecated(adata, t_range=(-2, 5), photostim_range=None):
    # Find frame numbers
    photostim_sequence = adata.uns['photostim_sequence']
    if photostim_range:
        photostim_sequence = photostim_sequence.iloc[photostim_range[0]:photostim_range[1]]
    trig = an.trigger_inds(adata.obs, trigger_t=photostim_sequence['t'], t_range=t_range, safe=False)
    ind0 = np.where(adata.uns['triggers']['scan2vr_idx'])[0][0] # first frame after VR start
    idyx = trig['idyx'] + ind0 + 1 # +1 for matlab indexing
    assert len(photostim_sequence) == len(idyx)

    frames_mat = os.path.join(adata.uns['path']['preprocessed_dir'], 'photostim', 'photostim_frames_{label}.mat')
    session_mat = os.path.join(adata.uns['path']['raw2P_dir'], 'session.mat')

    if adata.uns['ops']['slm_photostim']:
        groups = sorted(photostim_sequence['group'].unique())
        idys = {f'group{group}': np.array(photostim_sequence['group'] == group) for group in groups}
    else:
        rois = sorted(photostim_sequence['roi'].unique())
        idys = {f'roi{roi}': np.array(photostim_sequence['roi']== roi) for roi in rois}
    
    # Export average frame for each stim group
    for label, idy in idys.items():
        sess.export_mat_frames(idyx[idy], trig['t'], session_mat, frames_mat.format(label=label))
        idy_shuffle = idy.copy()
        np.random.shuffle(idy_shuffle)
        sess.export_mat_frames(idyx[idy_shuffle], trig['t'], session_mat, frames_mat.format(label=label + '_shuffle'))

def export_photostim_average_frames(adata, t_range=(-2, 5), photostim_sequence=None):
    if photostim_sequence is None:
        photostim_sequence = adata.uns['photostim_sequence']
    trig = an.trigger_inds(adata.obs, trigger_t=photostim_sequence['t'], t_range=t_range, safe=False)
    ind0 = np.where(adata.uns['triggers']['scan2vr_idx'])[0][0] # first frame after VR start
    idyx = trig['idyx'] + ind0 + 1 # +1 for matlab indexing
    assert len(photostim_sequence) == len(idyx)

    frames_mat = os.path.join(adata.uns['path']['preprocessed_dir'], 'photostim', 'photostim_frames_{label}.mat')
    session_mat = os.path.join(adata.uns['path']['raw2P_dir'], 'session.mat')

    if adata.uns['ops']['slm_photostim']:
        groups = sorted(photostim_sequence['group'].unique())
        group_idys = {f'group{group}': np.array(photostim_sequence['group'] == group) for group in groups}
        
        for label, idy in group_idys.items():
            sess.export_mat_frames(idyx[idy], trig['t'], session_mat, frames_mat.format(label=label))

    # rois = sorted(photostim_sequence['roi'].unique())
    # roi_idys = {f'roi{roi}': np.array(photostim_sequence['roi']== roi) for roi in rois}
    # for label, idy in roi_idys.items():
        # sess.export_mat_frames(idyx[idy], trig['t'], session_mat, frames_mat.format(label=label))

def compute_dF_image(adata, roi, tlim0=(-1, 0), tlim=(0, 1), signal='avgMov'):
    mat = scipy.io.loadmat(adata.uns['path']['photostim_frames'].format(condition=f'roi{roi}'))
    frames = mat[signal]
    t = mat['t'][0]
    frames = np.moveaxis(frames, 2, 0)
    frames -= frames.min()
    F0 = frames[(t>=tlim0[0]) & (t<tlim0[1])].mean(axis=0)
    F = frames[(t>=tlim[0]) & (t<tlim[1])].mean(axis=0)
    dF = (F-F0) / F0
    return dF

def compile_dF_images(adata, tlim0=[-1, 0], tlim=[0, 1], signal='avgMov'):
    adata.uns['photostim_dF_images'] = {}
    adata.uns['photostim_dF_images']['tlim0'] = tlim0
    adata.uns['photostim_dF_images']['tlim'] = tlim
    adata.uns['photostim_dF_images']['signal'] = signal
    adata.uns['photostim_dF_images']['images'] = {}
    for roi in adata.uns['photostim_targets']['roi'].unique():
        print(f'Computing dF image for roi {roi}.')
        dF = compute_dF_image(adata, roi, tlim0=tlim0, tlim=tlim, signal=signal)
        adata.uns['photostim_dF_images']['images'][f'roi{roi}'] = dF

def group2roi_dict(adata):
    targets = adata.uns['photostim_targets']
    group2roi = {}
    groups = [1, 2]
    for group in groups:
        group2roi[group] = sorted(targets.loc[targets['group']==group, 'roi'].unique())
    return group2roi

def cell_dF_image(adata, var_name, rois, crop=(50, 50)):
    group2roi = group2roi_dict(adata)
    dF = {}
    groups = [1, 2]
    for group in groups:
        group_rois = [roi for roi in rois if roi in group2roi[group]]
        dF_images = [adata.uns['photostim_dF_images']['images'][f'roi{roi}'] for roi in group_rois]
        if len(dF_images) == 0:
            return None
        dF_mean = np.stack(dF_images).mean(axis=0)
        dF[group] = dF_mean
    diff = dF[1] - dF[2] # group1 - group2

    stati = sess.fetch_cell_stat(adata, var_name)
    diff_crop = fc.crop_img(diff, stati['med'], crop=crop)
    if diff_crop.shape == crop:
        return diff_crop
    else:
        return None

