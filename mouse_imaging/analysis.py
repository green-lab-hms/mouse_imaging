import itertools
import pandas as pd
import numpy as np
from numpy import pi
from mouse_imaging import functions as fc
import igraph
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import warnings
import scipy.stats
import scipy.interpolate
import scipy.ndimage
import scipy.signal
import anndata
import time
from tqdm import tqdm
import random

import importlib
importlib.reload(fc)

# Select adatas
def select_metadata(adatas, key):
    for k, v in key.items():
        adatas = [adata for adata in adatas if adata.uns['metadata'][k]==v]
    return adatas

# Order cells
def order_cells_bybin(adata, obs, bins=20):
    df = binX1(adata, obs=obs, bins=bins)
    cell_order = df.idxmax(axis=1).sort_values().index
    return cell_order

def order_cells_bytime(adata, trigger, t_range=(-5, 10)):
    Xtrig, t = triggerX(adata, trigger=trigger, t_range=t_range)
    Xtrig_trialmean = np.nanmean(Xtrig, axis=1)
    peak_idx = np.argmax(Xtrig_trialmean, axis=1)
    cell_order = adata.var_names[np.argsort(peak_idx)]
    return cell_order

def order_cells_by_yiti(adata, layer='dcnv_norm', require_correct=True):
    if require_correct:
        adata = adata[adata.obs['correct']]
    X_y = binX1(adata[adata.obs['inITI']==0], obs='y', bins=20)
    trig = trigger_inds(adata.obs, trigger_name='inITI', t_range=(0, 3), min_dist_s=0, direction='rising', safe=True)
    X_iti_trials = fetch_X(adata.layers['dcnv_norm'], trig['idyx'])
    # X_iti_trials, t = triggerX(adata, trigger='inITI', t_range=(0, 3))
    X_iti = np.nanmean(X_iti_trials, axis=1)
    X_y_iti = np.concatenate([X_y, X_iti], axis=1)
    peak_idx = np.argmax(X_y_iti, axis=1)
    cell_order = adata.var_names[np.argsort(peak_idx)]
    return cell_order

def split_order(adata, split, order=order_cells_bybin, **order_kwargs):
    cell_order = [order(adata[:, split==split_category], **order_kwargs) for split_category in split.dtype.categories]
    cell_order = np.concatenate(cell_order)
    return cell_order

# Add variables
def downsample_reward_vr_to_obs(adata):
    vr_reward = np.zeros(len(adata.obs))
    reward_vr_inds = fc.rising_idx(adata.uns['vr']['reward'])
    reward_ts = adata.uns['vr']['t'][reward_vr_inds]
    for t in reward_ts:
        inds = np.where(adata.obs['t'] >= t)[0]
        if len(inds) > 0:
            vr_reward[inds[0]] = 1
    return vr_reward

def get_stem_length(vr):
    ymax = vr['y'].max()
    if ymax < 100:
        stem_length = 86
    elif ymax < 200:
        stem_length = 150
    elif ymax > 300:
        stem_length = 300
    return stem_length

# Graph processing
def sparsematrix_to_graph(matrix):
    sources, targets = matrix.nonzero()
    edgelist = zip(sources.tolist(), targets.tolist())
    g = igraph.Graph(edgelist)
    w = np.array(matrix[sources, targets])[0]
    return g, w

# Call cell types
def threshold_channel(var, threshold):
    for chan, thresh in threshold.items():
        var[f'{chan}+'] = var[chan] >= thresh
    var['celltype'] = var.apply(label_cell, axis=1).astype('category')

def metric_vs_cutoff(adata, test_channel, ref_channel, maxval=None, metric='specificity'):
    var = adata.var.copy()
    var = var[var[test_channel + '_spatial_corr'] > 0.5]
    cutoff = np.arange(0, 1, 0.05).astype(np.float64)
    
    maxval_computed = var[test_channel].dropna().sort_values()[-3:].mean()
    if maxval is None: 
        maxval = maxval_computed
    print('max value: %.0f' %maxval_computed)
    sr = pd.Series(index=cutoff)
    sr.index.name = 'cutoff'
    for cutoffi in cutoff:
        cells = var[var[test_channel] >= cutoffi * maxval]
        if metric == 'specificity':
            sr[cutoffi] = cells[ref_channel+'+'].mean()
        elif metric == 'count':
            sr[cutoffi] = len(cells)
        else:
            raise ValueError(f'Unsupported metric {metric}.')
    return sr

# Bin data
def specify_bins(sr, bins=None):
    if bins is None and pd.api.types.is_categorical(sr):
        bins = sorted(sr.unique())
    else:
        bins = pd.cut(sr, bins=bins).dtype.categories
    return bins

def bin_data(data, by, bins=20, metric=np.nanmean):
    binned = pd.cut(by, bins=bins)
    sr = pd.Series(index=binned.dtype.categories)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for bini in binned.dtype.categories:
            sr.loc[bini] = np.nanmean(data[binned == bini])
    return sr

def _binX1_deprecated(adata, obs, bins=20, layer='dcnv_norm'):
    assert obs is not None
    obs_binned = pd.cut(adata.obs[obs], bins=bins)
    bin_intervals = obs_binned.dtype.categories
    df = pd.DataFrame(index=adata.var_names, columns=bin_intervals)
    if layer == 'X':
        X = adata.X
    else:
        X = adata.layers[layer]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for bini in bin_intervals:
            df.loc[:, bini] = X[obs_binned == bini, :].mean(axis=0)
    return df

def _binX1(adata, obs, bins=20, layer='dcnv_norm', statistic=np.mean):
    assert obs is not None
    df = adata.to_df(layer=layer)
    df['cut'] = pd.cut(adata.obs[obs], bins=bins)
    df_out = df.groupby('cut', observed=False)[adata.var_names].apply(lambda x: statistic(x, axis=0))
    return df_out.T

def binX1(adata, obs=None, bins=None, obs_key=None, var_key=None, norm=False):
    obs_idx = fetch_index(adata.obs, obs_key)
    var_idx = fetch_index(adata.var, var_key)
    df = _binX1(adata[obs_idx, var_idx], obs=obs, bins=bins)
    if norm:
        _df = _binX1(adata[:, var_idx], obs=obs, bins=bins)
        cell_min = _df.min(axis=1)
        cell_max = _df.max(axis=1)
        df = df.subtract(cell_min, axis=0).divide(cell_max-cell_min, axis=0)
    return df

def binX2(adata, y, x, ybins=None, xbins=None):
    xbins = specify_bins(adata.obs[x], xbins)
    ybins = specify_bins(adata.obs[y], ybins)
    y_binned = pd.cut(adata.obs[y], bins=ybins)
    y_is_categorical = pd.api.types.is_categorical_dtype(adata.obs[y])

    df = pd.DataFrame(index=ybins, columns=xbins, dtype=np.float64)
    for ybin in ybins:
        if y_is_categorical:
            idy = adata.obs[y] == ybin
        else:
            idy = y_binned == ybin
        df.loc[ybin] = binX1(adata[idy], x, bins=xbins).mean(axis=0)
    return df

def fetch_X(X, idyx, norm=False):
    Xtrig = np.zeros((X.shape[1], *idyx.shape)) # axes=(cells, epochs, time)
    Xtrig[:] = np.nan
    for i, cell in enumerate(X.T):
        Xtrig[i] = fc.index_safe(cell.flatten().copy(), idyx)
    if norm:
        Xtrig = norm_Xtrig_cells(Xtrig)
    return Xtrig

def norm_Xtrig_cells(Xtrig):
    Xtrig = Xtrig.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Xtrig_trialmean = np.nanmean(Xtrig, axis=1)
        Xtrig -= Xtrig_trialmean.min(axis=-1)[:, None, None]
        Xtrig /= Xtrig_trialmean.max(axis=-1)[:, None, None]
    return Xtrig

def fetch_df(df, idy):
    df2 = pd.DataFrame(columns=df.columns)
    for col in df.columns:
        df2[col] = df[col].iloc[idy]
    return df2

def cell_bin_statistic(df, idxmax, idxmin, statistic=scipy.stats.ranksums):
    var_names = df.drop('cut', axis=1).columns
    sr = pd.Series(index=var_names, dtype='float64')
    for var_name in var_names:
        vals = []
        for idx in [idxmax, idxmin]:
            obs_idx = df['cut'] == idx[var_name]
            vals_i = df.loc[obs_idx, var_name]
            vals.append(vals_i)
        sr[var_name] = statistic(*vals).pvalue
    return sr    
        
def cell_tuning(adata, obs_col, obs_key=None, bins=20, min_delta=None, min_ratio=None, min_pvalue=None, layer='dcnv_norm'):
    # Select time
    obs_idx = fetch_index(adata.obs, obs_key)
    adata = adata[obs_idx]
    
    # Compute binned df
    df = adata.to_df(layer=layer)
    df['cut'] = pd.cut(adata.obs[obs_col], bins=bins, ordered=True)
    df_bin = df.groupby('cut', observed=False)[adata.var_names].mean().T
    df_bin.columns = df['cut'].dtype.categories
    cell_bin = np.array(df_bin.columns.mid[np.argmax(df_bin.values, axis=1)])

    max_val = df_bin.max(axis=1)
    min_val = df_bin.min(axis=1)

    if min_delta is not None:
        delta = max_val - min_val
        cell_bin[delta < min_delta] = np.nan

    if min_ratio is not None:
        ratio = max_val / min_val
        cell_bin[ratio < min_ratio] = np.nan

    if min_pvalue is not None:
        idxmax = df_bin.idxmax(axis=1)
        idxmin = df_bin.idxmin(axis=1)
        pvalue = cell_bin_statistic(df, idxmax, idxmin, statistic=scipy.stats.ranksums)
        cell_bin[pvalue > min_pvalue] = np.nan
    
    cell_bin = pd.Series(cell_bin, index=adata.var_names)
        
    return cell_bin

def match_index(df, x, match, xbins=None, matchbins=None, equal_weight=False):
    df = df.copy()
    matchcut = pd.cut(df[match], bins=matchbins)
    matchbins = matchcut.dtypes.categories

    df['cut'] = pd.cut(df[x], bins=xbins)
    xbins = df['cut'].dtypes.categories

    # Weigh sampling per bin
    if equal_weight:
        weight = np.ones(len(xbins))
    else:
        weight = df.groupby('cut')[x].count()
        weight /= weight.sum()

    cutinds = df['cut'].dtypes.categories

    inds = pd.Index([])
    for matchbin in matchbins:
        dfi = df[matchcut==matchbin]
        grouped = dfi.groupby('cut')
        n_samples = grouped.apply(lambda x: len(x))
        if n_samples.min() == 0: continue

        if equal_weight:
            inds = inds.union( grouped.sample(n=n_samples.min(), replace=False).index )
        else:
            limiting_bin = np.argmin(n_samples / weight)
            weighti = ( (weight / weight[limiting_bin]) * (n_samples[limiting_bin]) ).astype(int)
            for xbin in xbins:
                dfij = dfi[dfi['cut']==xbin].sample(n=weighti[xbin], replace=False)
                inds = inds.union(dfij.index)
    return inds
    
# Regression
def activity_regression(adata, predictors, model, obs_key=None):
    obs_idx = fetch_index(adata.obs, obs_key)
    adatai = adata[obs_idx, :]

    # Predictors
    X = adatai.obs[predictors]
    X = StandardScaler().fit_transform(X)

    # Activity
    Y = adatai.layers['dcnv_norm'].copy()
    Y = StandardScaler().fit_transform(Y)

    model.fit(X, Y)
    df = pd.DataFrame(model.coef_, index=adata.var_names, columns=predictors)
    
    return df

# Resampling cell populations
def resample_celltypes_to_min_size(adata, celltypes, n_resample=100, layer='dcnv_norm'):
    # Pre-allocate new columns
    columns = [f'{name}_{layer}_subsample_{i_resample}' for name, celltype in celltypes.items() for i_resample in range(n_resample) ]
    df_append = pd.DataFrame(index=adata.obs.index, columns=columns, dtype='float64')
    adata.obs = pd.concat([adata.obs, df_append], axis=1)
    
    # Populate new columns
    n_cells = {name:(adata.var['celltype']==celltype).sum() for name, celltype in celltypes.items()}
    n_min = min(n_cells.values())
    for name, celltype in tqdm(celltypes.items()):
        var_names = adata[:, adata.var['celltype']==celltype].var_names
        for i_resample in range(n_resample):
            inds = random.sample(range(n_cells[name]), k=n_min)
            idx = var_names[inds]
            adata.obs[f'{name}_{layer}_subsample_{i_resample}'] = adata[:, idx].layers[layer].mean(axis=1)

# Trigger data
def trigger_inds(df, trigger_name=None, trigger_idy=None, trigger_t=None, dt=None, t_range=(-5, 10), min_dist_s=0, direction='rising', offset_s=0, safe=False):
    if dt is None:
        dt = df['dt'].mean()
    if trigger_idy is not None:
        idy = trigger_idy
    elif trigger_t is not None:
        idy = np.array([np.where(df['t'] >= ti)[0][0] for ti in trigger_t])
    elif trigger_name is not None:
        min_dist = min_dist_s / dt
        idy = fc.rising_idx(df[trigger_name], min_dist=min_dist, direction=direction)
    _idyx, t = fc.idyx(idy, t_range, dt)
    offset = int(offset_s / dt)
    idy_offset = idy + offset
    if safe:
        safe_idx = (_idyx >= 0).all(axis=1) & (_idyx < len(df)).all(axis=1)
        idy = idy[safe_idx]
        _idyx = _idyx[safe_idx]
        idy_offset = idy_offset[safe_idx]
    trig = dict(idy=idy, idyx=_idyx, t=t, idy_offset=idy_offset, ntrials=len(idy), t_range=t_range)
    return trig

def trigger_df(df, layers, obs_cols=None, **trig_kwargs):
    # adata = adata.copy()
    
    # Define trigger, out-of-bounds triggers removed at this point
    trig = trigger_inds(df, safe=True, **trig_kwargs)
    
    # Compute triggered data
    layer_vals = {
        col: df[col].values[trig['idyx']] for col in layers
    }
    
    obs = pd.DataFrame(index=np.arange(trig['ntrials']))
    for col in obs_cols:
        obs[col] = df[col].values[trig['idy_offset']]
    obs.index = obs.index.astype(str) # for AnnData index
        
    # Compute time-associated data
    var = pd.DataFrame({'t': trig['t']})
    var.index = var.index.astype(str) # for AnnData index
    
    # Define unstructured data
    uns = dict(trig=trig)
        
    trig_adata = anndata.AnnData(X=trig['idyx'], var=var, obs=obs, uns=uns, layers=layer_vals)
    return trig_adata

def trigger_dfs(dfs, **kwargs):
    # Check that frame rate is the same
    assert (np.diff([df['dt'].mean() for df in dfs]) < 0.01).all()
    
    assert 'dt' not in kwargs.keys()
    dt = dfs[0]['dt'].mean()
    
    trig_cat = anndata.concat([trigger_df(df, dt=dt, **kwargs) for df in dfs], merge='first', index_unique='-')
    trig_cat.obs_names_make_unique()
    return trig_cat

def trigger_adata(adata, layers, obs_cols=None, **trig_kwargs):

    trig_adata = trigger_df(adata.obs, layers, obs_cols=obs_cols, **trig_kwargs)
    return trig_adata

def _trigger_adatas(adatas, **kwargs):
    """
    Returns: anndata
    X: (epochs, time)
    obs: (epochs,)
    var: (time,)
    """
    # Check that frame rate is the same
    assert (np.diff([adata.obs['dt'].mean() for adata in adatas]) < 0.01).all()
    
    assert 'dt' not in kwargs.keys()
    dt = adatas[0].obs['dt'].mean()
    
    trig_cat = anndata.concat([trigger_adata(adata, dt=dt, **kwargs) for adata in adatas], axis=0, merge='first', index_unique='-')
    trig_cat.obs_names_make_unique()
    return trig_cat

def trigger_adatas(adatas, layers=None, subsample=False, n_resample=100, **kwargs):
    if subsample:
        print("Triggering on subsampled data.")
        for i_resample in tqdm(range(n_resample)):
            # Define anndata layers that are subsampled
            layers_i = [f'{layer}_{i_resample}' if layer.endswith('subsample') else layer for layer in layers]
            layers_subsample_i = [layer for layer in layers_i if ('subsample' in layer)]
            
            # Trigger on subsampled data
            trigi = _trigger_adatas(adatas, layers=layers_i, **kwargs)
            
            if i_resample == 0:
                # If first iteration, initiate trigger object
                trig = trigi.copy()
                layers_subsample = ['_'.join(layer.split('_')[:-1]) for layer in layers_subsample_i]
                # Initialize layer without resample num at zero
                for layer in layers_subsample:
                    trig.layers[layer] = np.zeros(trig.shape)
                
                # Remove layer_0
                for layer in layers_subsample_i:
                    del trig.layers[layer]
            
            # Add subsampled layers to initial layer
            for layer in layers_subsample:
                layer_i = f'{layer}_{i_resample}'
                trig.layers[layer] += trigi.layers[layer_i]
                    
        # Once done adding, divide by number of resamples
        for layer in layers_subsample:
            trig.layers[layer] /= n_resample
    else:
        trig = _trigger_adatas(adatas, layers=layers, **kwargs)
    return trig

def trigger_adatas_mean(mice, layers, obs_cols=None, **trig_kwargs):
    if obs_cols != set():
        raise IOError('Function does not support obs_cols yet.')
        
    # Trigger adatas for each list separately
    trigs = []
    for mouse in mice:
        trig = trigger_adatas(mouse, layers=layers, obs_cols=obs_cols, **trig_kwargs)
        # idy = fetch_index(trig.obs, obs_key)
        # trig = trig[idy]
        trigs.append(trig)
    
    # Truncate such that they have the same length
    min_trials = min(map(len, trigs))
    trigs = [trig[:min_trials] for trig in trigs]
    
    # Take mean across adatas
    trig_mean = trigs[0]
    for layer in trigs[0].layers.keys():
        trig_mean.layers[layer] = np.nanmean([trig.layers[layer] for trig in trigs], axis=0)
    return trig_mean

def get_tlim_idx(t, tlim):
    idt = (t >= tlim[0]) & (t < tlim[1])
    return idt

def trigger_cells_adata(adata, activity_layer, obs_cols=None, tlim0=None, tlim=None, **trig_kwargs):
    """
    Returns: anndata
    X: (epochs, cells)
    obs: (epochs,)
    var: (cells,)
    """
    
    # Define trigger, out-of-bounds triggers removed at this point
    trig = trigger_inds(adata.obs, safe=True, **trig_kwargs)
    
    # Compute triggered data
    Xtrig = fetch_X(adata.layers['dcnv_norm'], trig['idyx'])
    idt = get_tlim_idx(trig['t'], tlim)
    idt0 = get_tlim_idx(trig['t'], tlim0)
    dXtrig = Xtrig[:, :, idt].mean(axis=-1)
    if tlim0 is not None:
        dXtrig -= Xtrig[:, :, idt0].mean(axis=-1)
    # layer_vals = {activity_layer: dXtrig.T}
    
    obs = pd.DataFrame(index=np.arange(trig['ntrials']))
    for col in obs_cols:
        obs[col] = adata.obs[col].values[trig['idy_offset']]
    obs.index = obs.index.astype(str) # for AnnData index
        
    # Compute time-associated data
    var = pd.DataFrame({'cell': adata.var_names})
    var.index = var.index.astype(str) # for AnnData index
    
    # Define unstructured data
    uns = dict(trig=trig)
        
    trig_adata = anndata.AnnData(X=dXtrig.T, var=var, obs=obs, uns=uns,)
    return trig_adata

def assert_var_names_are_equal(adatas):
    var_names = adatas[0].var_names
    for adata in adatas[1:]:
        assert (adata.var_names == var_names).all()

def trigger_cells_adatas(adatas, **kwargs):
    """
    Returns: anndata
    X: (epochs, time)
    obs: (epochs,)
    var: (time,)
    """
    # Check that frame rate is the same
    assert (np.diff([adata.obs['dt'].mean() for adata in adatas]) < 0.01).all()
    
    assert 'dt' not in kwargs.keys()
    dt = adatas[0].obs['dt'].mean()

    assert_var_names_are_equal(adatas)
    
    trig_cat = anndata.concat([trigger_cells_adata(adata, dt=dt, **kwargs) for adata in adatas], axis=0, merge='first', index_unique='-')
    return trig_cat

def select_triggered_data(trig, obs_keys_set):
    data = {}
    for j, obs_keys in enumerate(obs_keys_set):
        data[j] = {}
        # Loop through lines
        for k, obs_key in enumerate(obs_keys):
            idy = fetch_index(trig.obs, obs_key)
            data[j][k] = trig[idy].X

    d = {}
    for k in range(len(obs_keys)):
        datajk = np.concatenate([data[0][k], data[1][k]], axis=0)
        print(f'{k}: {len(datajk)} trials')
        d[k] = datajk
    return d    

def flip_trig_rows(trig, flip_key):
    if flip_key is None:
        return trig    
    trig = trig.copy()
    for key, layers in flip_key.items():
        for layer in layers:
            trig.layers[layer][trig.obs['h_pulse_left']] *= -1
    return trig

def X_time_diff(adata, t, layer='dcnv_norm', tlim=(0.5, 1.5), tlim0=(-1, 0)):
    if tlim0 is None:
        t_range = tlim
    else:
        t_range = (tlim0[0], tlim[1])
        
    trig = trigger_inds(adata.obs, trigger_t=t, t_range=t_range, safe=False)
    if layer == 'X':
        X = adata.X
    else:
        X = adata.layers[layer]
    Xtrig = fetch_X(X, trig['idyx'])
    
    X1 = Xtrig[:, :, (trig['t'] >= tlim[0]) & (trig['t'] < tlim[1])].mean(axis=2)
    if tlim0 is None:
        return X1
    else:
        X0 = Xtrig[:, :, (trig['t'] >= tlim0[0]) & (trig['t'] < tlim0[1])].mean(axis=2)
        dX = X1 - X0
        return dX

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

def activity_peak_ts(adata, signal):
    widths_s = np.arange(0.2, 3, 0.2)
    widths = widths_s / adata.obs['dt'].mean()
    inds = scipy.signal.find_peaks_cwt(adata.obs[signal], widths)
    peak_vals = adata.obs[signal][inds].sort_values(ascending=False)
    peak_inds = peak_vals.index
    peak_inds = peak_inds[adata.obs['inITI'].loc[peak_inds]==False]
    peak_ts = adata.obs['t'][peak_inds]
    return peak_ts

def h_trigger(h, thresh):
    trigger1 = fc.rising_idx(h > thresh)
    trigger1 = trigger1[np.abs(h)[trigger1]<np.pi/2]
    trigger2 = fc.rising_idx(h < -thresh)
    trigger2 = trigger2[np.abs(h)[trigger2]<np.pi/2]
    trigger = np.sort(np.concatenate([trigger1, trigger2]))
    trigger_idx = np.zeros(len(h)).astype(bool)
    trigger_idx[trigger] = True
    return trigger_idx

# Old Triggering functions
def index_trig(trig, idx):
    trig = trig.copy()
    trig['idy'] = trig['idy'][idx]
    trig['idyx'] = trig['idyx'][idx]
    trig['idy_offset'] = trig['idy_offset'][idx]
    return trig

def resample_trig(trig, df1, df2, dt2=None, **trig_kwargs):
    t_trig = df1['t'].iloc[trig['idy']]
    df2_idy = np.array([np.where(df2['t'] >= t0)[0][0] for t0 in t_trig])
    df2_trig = trigger_df(df2, trigger_idy=df2_idy, dt=dt2, **trig_kwargs)
    assert trig['ntrials'] == df2_trig['ntrials']
    return df2_trig

def trigger_obs_vr(adata, trigger_name=None, trigger_idy=None, trigger_t=None, dt={'obs': None, 'vr': None}, **trig_kwargs):
    obs_trig = trigger_df(adata.obs, trigger_name=trigger_name, trigger_idy=trigger_idy, trigger_t=trigger_t, dt=dt['obs'], **trig_kwargs)
    vr_trig = resample_trig(obs_trig, adata.obs, adata.uns['vr'], dt2=dt['vr'], **trig_kwargs)
    trig = {'obs': obs_trig, 'vr': vr_trig, 'rank_val': None, 'kwargs': trig_kwargs}
    return trig

def trig_X_mean(adata, trig, var_key, tlim=(0, 1)):
    adata = adata.copy()
    trig = trig.copy()
    idx_var = fetch_index(adata.var, var_key)
    Xtrig = fetch_X(adata[:, idx_var].X, trig['idyx'], norm=True)
    idt = (trig['t'] >= tlim[0]) & (trig['t'] < tlim[1])
    Xtrig_cellmean = np.nanmean(Xtrig[:, :, idt], axis=-1).mean(axis=0)
    return Xtrig_cellmean

def argsort_trig_by_X(adata, trig, var_key, tlim=(0, 1)):
    Xtrig_cellmean = trig_X_mean(adata, trig, var_key, tlim=(0, 1))
    order = np.argsort(Xtrig_cellmean)
    return order

def trigger_X_idx(adata, threshold=0.6, onset_threshold=0.25, var_key=None):
    var_idx = fetch_index(adata.var, var_key)
    Xmean = adata[:, var_idx].X.mean(axis=1)    

    onset_inds = fc.peak_onset(Xmean, peak_thresh=threshold, onset_thresh=onset_threshold)
    onset_arr = np.zeros(len(adata.obs)).astype(bool)
    onset_arr[onset_inds] = True
    return onset_arr

def epoch_idx(df, idy, key):
    df_epoch = df[key.keys()].iloc[idy]
    idx_epoch = fetch_index(df_epoch, key)
    return idx_epoch

def compute_Xtrig(adata, trig=None, layer='dcnv', obs_key=None, var_key=None, norm_cells=False):
    adata = adata.copy()
    idx_var = fetch_index(adata.var, var_key)
    Xtrig = fetch_X(adata[:, idx_var].layers[layer], trig['idyx'], norm=norm_cells)
    
    if obs_key is not None:
        idx_epoch = epoch_idx(adata.obs, trig['idy_offset'], obs_key)
        Xtrig = Xtrig[:, idx_epoch]
    return Xtrig

def compute_VRtrig(vr, col=None, trig=None, obs_key=None, sigma=0, nan_other_trials=False, nan_iti=False):
    VRtrig = fc.index_safe(vr[col].to_numpy(), trig['idyx'])

    if sigma: 
        VRtrig = gaussian_filter1d(VRtrig, sigma=sigma, axis=1)

    if nan_other_trials:
        trials_trig = fc.index_safe(vr['trial'].to_numpy(), trig['idyx'])
        curr_trial = vr['trial'].to_numpy()[trig['idy_offset']]
        trials_trig -= curr_trial[:, None]
        VRtrig[trials_trig != 0] = np.nan

    if nan_iti:
        iti_trig = fc.index_safe(vr['inITI'].to_numpy(), trig['idyx'])
        VRtrig[iti_trig == 1] = np.nan

    if obs_key is not None:
        idx_epoch = epoch_idx(vr, trig['idy_offset'], obs_key)
        VRtrig = VRtrig[idx_epoch]

    return VRtrig


# Splitting data
def key_to_figshape(shape):
    nrows = shape[0]
    ncols = 1 if len(shape) == 1 else shape[1]
    return nrows, ncols

def split_keys(df, split, constant=None, ret_shape=False):
    if type(split) is not list: 
        split = [split]

    values = [sorted(df[spliti].unique()) for spliti in split]
    values = [[value for value in valuesi if value != 'none'] for valuesi in values]
    value_combinations = list(itertools.product(*values))
    keys = [{keyi: vali for keyi, vali in zip(split, val_comb)} for val_comb in value_combinations]

    if constant:
        for key in keys:
            key.update(constant)

    if ret_shape:
        shape = [len(value) for value in values] # shape helps for formatting plotting
        nrows, ncols = key_to_figshape(shape)
        return keys, (nrows, ncols)
    else:
        return keys

def fetch_index(df, key):
    idx = np.ones(len(df)).astype(bool)
    if key is None:
        return idx

    for keyi, vali in key.items():
        if (type(vali) is str) and (vali[0] in ['<', '>', '=', '!']):
            arr = df[keyi]
            idx &= eval('arr' + vali)
        else:
            idx &= df.reset_index()[keyi] == vali
    if isinstance(idx, pd.Series):
        idx = idx.values
    return idx

def split_df(df, keys):
    if keys is None:
        yield df, None

    for key in keys:
        idx = fetch_index(df, key)
        yield idx, key

def split_metric(adata, split, _metric):
    df = pd.DataFrame()
    for adatai, combination in split_adata(adata, split):
        name =  _metric.__name__ + '; ' + str(combination).strip('{}').replace("'", '')
        df[name] = _metric(adatai)
    return df

# Aggregating data
def aggr_var(sessions):
    var = pd.concat([session.var for session in sessions], axis=0)
    var = var[var.index.str.contains('cell')].reset_index().drop('index', axis=1)
    return var

# Metrics
def iti_response(adata, t_before=(-5, -3), t_after=(0, 5)):
    t_ranges = np.concatenate([t_before, t_after])
    t_range = (t_ranges.min(), t_ranges.max())
    df = triggerX(adata, y='cell', trigger='inITI', t_range=t_range)
    t = df.columns
    idx_before = (t >= t_before[0]) & (t < t_before[1])
    idx_after = (t >= t_after[0]) & (t < t_after[1])
    result = df.iloc[:, idx_after].mean(axis=1) - df.iloc[:, idx_before].mean(axis=1)
    return result

def mean(adata):
    result = adata.X.mean(axis=0)
    return result

# Cross correlation
def crosscorr(adata, tlim=(-5, 5)):
    assert adata.shape[1] == 2
    X = adata.X
    X = X / X.std(axis=0)
    
    # Compute crosscorr
    Xlen = X.shape[0]
    xcorr = scipy.signal.correlate(*X.T) / Xlen
    
    # Compute time base
    lags = np.arange(-Xlen + 1, Xlen) * adata.obs['dt'].mean()
    assert len(lags) == len(xcorr)

    idx = (lags >= tlim[0]) & (lags < tlim[1])
    
    return xcorr[idx], lags[idx]
 
# Activity
def add_normalized_celltype_mean(adata):
    X = adata.X
    Fmax = np.percentile(X, 95, axis=0)
    active_cells = Fmax > 0.2
    X = X[:, active_cells]
    X = (X / Fmax[active_cells])
    Sst44_cells = adata.var['Sst44+'][active_cells]
    adata.obs['Sst44_norm'] = X[:, Sst44_cells].mean(axis=1)
    adata.obs['Non-labeled_norm'] = X[:, ~Sst44_cells].mean(axis=1)

def normalize_activity(X, nonzero_percentile=95):
    Xnonzero = X.copy()
    Xnonzero[Xnonzero==0] = np.nan
    Fmax = np.nanpercentile(Xnonzero, 95, axis=0)
    Xnorm = X / Fmax
    return Xnorm

# UMAP
def umap_X(X, observation_axis, **umap_params):
    assert observation_axis in ['cell', 'time']
    X_scaled = StandardScaler().fit_transform(X)
    if observation_axis == 'cell':
        X_scaled = X_scaled.T
    import umap # imported here because it is slow to import (~7 s)
    result = umap.UMAP(**umap_params).fit_transform(X_scaled)
    return result

def umap_Xtime(X, random_state=None, n_neighbors=15, min_dist=0.1, **umap_params):
    umap_params.update(dict(random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist))
    result = umap_X(X, 'time', **umap_params)
    return result

def umap_Xcell(X, random_state=None, n_neighbors=10, min_dist=0.1, **umap_params):
    umap_params.update(dict(random_state=random_state, n_neighbors=n_neighbors, min_dist=min_dist))
    result = umap_X(X, 'cell', **umap_params)
    return result

# Cell clustering 
def leiden_Xcell(X, n_neighbors=10, resolution=1, objective_function='CPM', n_iterations=-1):
    X_scaled = StandardScaler().fit_transform(X)
    M = kneighbors_graph(X_scaled.T, n_neighbors=n_neighbors, mode='distance', metric='euclidean')
    G, W = sparsematrix_to_graph(M)
    leiden = igraph.Graph.community_leiden(G, weights=W, resolution_parameter=resolution, 
                                           objective_function=objective_function, n_iterations=n_iterations)
    return leiden

def leiden_Xtime(X, n_neighbors=10, resolution=1, objective_function='CPM', n_iterations=-1):
    X_scaled = StandardScaler().fit_transform(X)
    M = kneighbors_graph(X_scaled, n_neighbors=n_neighbors, mode='distance', metric='euclidean')
    G, W = sparsematrix_to_graph(M)
    leiden = igraph.Graph.community_leiden(G, weights=W, resolution_parameter=resolution, 
                                           objective_function=objective_function, n_iterations=n_iterations)
    return leiden
