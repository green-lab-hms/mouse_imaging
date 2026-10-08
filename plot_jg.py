import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns
import numpy as np
import scipy.stats
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

from mouse_imaging.session import maze_id, unpack_var_name, fetch_cell_stat
import mouse_imaging.analysis as an
from mouse_imaging import functions as fc

from anndata._core.anndata import AnnData

import importlib
import copy
import warnings
import glob
import os
import textwrap
import itertools

from tifffile import imread, imwrite
from scipy.io import loadmat, savemat

importlib.reload(an)
importlib.reload(fc)

current_cmap = copy.copy(mpl.colormaps[mpl.rcParams["image.cmap"]])
current_cmap.set_bad(color='grey')

# Palettes
celltype_palette = {'B+': 'tab:blue',
                    'B+R+': 'tab:purple',
                    'R+': 'tab:red',
                    'Negative': 'silver'}
celltype_palette2 = {'B+R+': 'tab:blue',
                    'R+': 'tab:red',
                    'Negative': 'silver'}
trial_type_palette = {'black_left_correct': 'black', 
                      'black_left_incorrect': 'red',
                     'white_right_correct': 'dodgerblue',
                     'white_right_incorrect': 'magenta'}
trial_type_ITI_palette = {'black_left_correct': 'grey',
                          'black_left_correct_ITI': 'black',
                          'black_left_incorrect': 'red',
                          'black_left_incorrect_ITI': 'firebrick',
                          'white_right_correct': 'dodgerblue',
                          'white_right_correct_ITI': 'mediumblue',
                          'white_right_incorrect': 'magenta',
                         'white_right_incorrect_ITI': 'darkmagenta',
                         'black2_left_correct': 'grey',
                          'black2_left_correct_ITI': 'black',
                          'black2_left_incorrect': 'red',
                          'black2_left_incorrect_ITI': 'firebrick',
                          'white2_right_correct': 'dodgerblue',
                          'white2_right_correct_ITI': 'mediumblue',
                          'white2_right_incorrect': 'magenta',
                         'white2_right_incorrect_ITI': 'darkmagenta',
                         'trial_start': 'lime',
                         'in_intersection': 'coral'}
trial_type_switch_ITI_palette = {'black_left_correct': 'grey',
                          'black_left_correct_ITI': 'black',
                          'black_left_incorrect': 'red',
                          'black_left_incorrect_ITI': 'firebrick',
                          'white_right_correct': 'dodgerblue',
                          'white_right_correct_ITI': 'mediumblue',
                          'white_right_incorrect': 'magenta',
                          'white_right_incorrect_ITI': 'darkmagenta',
                          'black_white_right_correct': 'grey',
                          'black_white_right_correct_ITI': 'black',
                          'black_white_right_incorrect': 'red',
                          'black_white_right_incorrect_ITI': 'firebrick',
                          'white_black_left_correct': 'dodgerblue',
                          'white_black_left_correct_ITI': 'mediumblue',
                          'white_black_left_incorrect': 'magenta',
                          'white_black_left_incorrect_ITI': 'darkmagenta'}
iti_palette = {'notITI': 'grey', 'ITI_correct': 'green', 'ITI_incorrect': 'red'}

# Helper functions
def format_plot(ax, remove_spines=['top', 'right'], lw=0.25):
    for spine in remove_spines:
        ax.spines[spine].set_visible(False)
    if 'bottom' in remove_spines:
        ax.set_xticks([])
    if 'left' in remove_spines:
        ax.set_yticks([])
    ax.tick_params(width=lw)

def to_numerical_index(index, name):
    if name in ['cell', 'trial']:
        numerical = np.arange(len(index)+1)
    elif pd.api.types.is_interval_dtype(index):
        numerical = [bini.left for bini in index] + [index[-1].right]
    else:
        warnings.warn('Converting index to numerical range.')
        numerical = np.arange(len(index))
    return numerical

def session_title(data):
    if type(data) is AnnData:
        adata = data
        session_txt = '{mouse} {date} {region}'.format(**adata.uns['metadata'])
        neurons_txt = adata.var_names[0] if adata.n_vars == 1 else '%i neurons' %adata.n_vars
        title_ = '; '.join([session_txt, neurons_txt]) + '\n'
    elif type(data) is pd.core.frame.DataFrame:
        df = data
        maze_txt = df.attrs['maze']
        session_txt = '{mouse} {date} {session}'.format(**df.attrs)
        line1 = session_txt
        line2 = maze_txt
        title_ = '\n'.join([line1, line2])
    return title_

def sessions_title(adatas, wrap=40, trigger_name=None):
    mouse_ids = sorted(set([adata.uns['metadata']['mouse'] for adata in adatas]))
    regions = set([adata.uns['metadata']['region'] for adata in adatas])
    title = '; '.join([', '.join(mouse_ids),  ', '.join(regions), ])
    if trigger_name is not None:
        title = ', '.join([title, f'Triggered on {trigger_name}'])
    
    title = textwrap.fill(title, 40)
    return title

def trial_title(adata, trial):
    world = adata.uns['trials'].loc[trial, 'world']
    correctness = 'correct' if adata.uns['trials'].loc[trial, 'correct'] else 'incorrect'
    reward = '+reward' if adata.uns['trials'].loc[trial, 'rewarded_trial'] else '-reward'
    title = f'trial {trial}; {world} {correctness} {reward}'
    return title

def get_key_ntrials(adata, key):
    adatai = adata[an.fetch_index(adata.obs, key)]
    ntrials = len(adatai.obs.trial.unique())
    return ntrials

def key_title(key, adata=None):
    title = '\n'.join([f'{keyi}: {vali}' for keyi, vali in key.items()])
    if type(adata) is AnnData:
        ntrials = get_key_ntrials(adata, key)
        title += f'\n{ntrials} trials'
    return title

def add_vlines(ax, lines, **kwargs):
    for line in lines:
        ax.axvline(line, **kwargs)

def add_hlines(ax, lines, **kwargs):
    for line in lines:
        ax.axhline(line, **kwargs)

def datalim(arr, margin=0.2):
    lim = arr.min(), arr.max()
    mean = np.mean(lim)
    rng = lim[1] - lim[0]
    halfwidth = rng / 2 * (1 + margin)
    datalim = (mean - halfwidth, mean + halfwidth)
    return datalim

def mesh_ticks(ticks):
    ticks = np.array(ticks)
    dt = np.diff(ticks).mean()
    tmesh = np.insert(ticks+dt/2, 0, ticks[0]-dt/2)
    return tmesh

def update_gsi(gsi):
    if gsi is None:
        gs = mpl.gridspec.GridSpec(nrows=1, ncols=1)
        gsi = gs[0, 0]
    return gsi

# Basic plotting wrapper functions
def pcolormesh(arr2d, index=None, xticks=None, cbar=True, cbar_kwargs={}, **kwargs):
    yticks = np.arange(len(arr2d)+1)
    if xticks is None:
        if index is None:
            index = np.arange(arr2d.shape[1])
        xticks = mesh_ticks(index)
    m = np.ma.masked_where(np.isnan(arr2d),arr2d)
    plt.pcolormesh(xticks, yticks, m, **kwargs)
    if cbar:
        default_cbar_kwargs = dict(shrink=0.5, aspect=10)
        default_cbar_kwargs.update(cbar_kwargs)
        plt.colorbar(**default_cbar_kwargs)

def lineplot(arr2d, xticks=None, statistic=np.nanmean, label=None, ylim=None, legend_loc='best', show_ci=True, **plot_kwargs):
    assert len(arr2d.shape) == 2

    if xticks is None:
        xticks = np.arange(arr2d.shape[1])

    y_est = statistic(arr2d, axis=0)
    alpha = plot_kwargs.pop('alpha', 1)
    plt.plot(xticks, y_est, label=label, alpha=alpha,  **plot_kwargs)

    if show_ci and len(arr2d) > 1:
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                bootstrap_result = scipy.stats.bootstrap((arr2d,), statistic, axis=0, confidence_level=0.95, n_resamples=100, batch=100, method='basic')
                y_err = np.array(bootstrap_result.confidence_interval)
        except ValueError:
            y_err = np.zeros((2, arr2d.shape[1]))
            y_err[:] = np.nan
        plt.fill_between(xticks, y_err[0], y_err[1], alpha=0.2*alpha, **plot_kwargs)
    
    if not label is None:
        plt.legend(loc=legend_loc)

    if ylim is not None:
        plt.ylim(ylim)

def colored_line(x, y, c, cmap=None, show_colorbar=False, vmin=None, vmax=None, ax=None):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create a continuous norm to map from data points to colors
    if vmin is None:
        vmin = c.min()

    if vmax is None:
        vmax = c.max()
    norm = plt.Normalize(vmin, vmax)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    
    # Set the values used for colormapping
    lc.set_array(c)
    if ax is None:
        ax = plt.gca()
    line = ax.add_collection(lc)
    if show_colorbar:
        plt.colorbar(line, ax=ax)
    return line

# Plot nearest neighbor graphs
def nn_df1(adata, celltypes, neighbor_celltype, shuffle=False):
    NN = adata.varp['nearest_neighbor_10_dcnv_0.25sigma_norm'].copy()
    if shuffle:
        shuffle_idx = np.arange(NN.shape[1])
        np.random.shuffle(shuffle_idx)
        NN = NN[:, shuffle_idx]
    var0_idx = (adata.var['celltype'] == neighbor_celltype)
    neighbors = NN[:, var0_idx].mean(axis=1)
    
    df = pd.DataFrame(columns=['Fraction Sst44 neighbors', 'celltype', 'region', 'mouse'])
    rows = []
    for celltype in celltypes:
        var_idx = (adata.var['celltype'] == celltype)
        # var_scramble_idx = var_idx.sample(frac=1, replace=False)
        row = pd.Series({
            'Neighbor celltype fraction': neighbors[var_idx].mean(),
              'celltype': celltype,
              'region': adata.uns['metadata']['region'],
              'mouse': adata.uns['metadata']['mouse'],})
        rows.append(row)
    df = pd.concat(rows, axis=1).T
    return df

def nn_df(adatas, celltypes, neighbor_celltype, shuffle=False, region=None):
    df = pd.concat([nn_df1(adata, celltypes, neighbor_celltype, shuffle=shuffle) for adata in adatas], axis=0)
    df = df.groupby(['mouse', 'region', 'celltype'], as_index=False)['Neighbor celltype fraction'].mean()
    if region is not None:
        df = df[df['region']==region]
    df['celltype'] = pd.Categorical(df['celltype'], categories=['B+R+', 'R+', 'Negative'], ordered=True)
    df['region'] = pd.Categorical(df['region'], categories=['PPC', 'RSC'], ordered=True)
    return df

def plot_nearest_neighbor_fraction(adatas, neighbor_celltype, region, figsize=None, s=5):
    adatas = an.select_metadata(adatas, {'region': region})
    n_mice = len(np.unique([adata.uns['metadata']['mouse'] for adata in adatas]))
    print(f'{n_mice} mice, {len(adatas)} sessions')
    
    celltypes = ['B+R+', 'R+', 'Negative',]
    nn_stats = nn_df(adatas, celltypes, neighbor_celltype, shuffle=False, region=region)
    nn_stats_shuffle = nn_df(adatas, celltypes, neighbor_celltype, shuffle=True, region=region)
    neighbors_shuffle = nn_stats_shuffle['Neighbor celltype fraction'].mean()
    
    stats_test_nearest_neighbor(nn_stats, celltypes, comparison='B+R+')
    
    plt.figure(figsize=figsize)
    ax = plt.subplot()
    # plt.ylim(0, 0.21)
    sns.barplot(data=nn_stats, x='celltype', y='Neighbor celltype fraction', errwidth=0.25, palette=celltype_palette2)
    sns.swarmplot(data=nn_stats, x='celltype', y='Neighbor celltype fraction', color='black', s=s)
    ax.axhline(neighbors_shuffle, c='black', lw=0.25, ls='--', dashes=[10, 10])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks([])
    ax.tick_params(axis='y', which='both', width=0.25)
    
    # plt.legend()
    
def stats_test_nearest_neighbor(df, celltypes, comparison):
    sr0 = df[df['celltype']== comparison].set_index('mouse')['Neighbor celltype fraction'] 
    for celltype in celltypes:
        if celltype == comparison: continue
        sr1 = df[df['celltype']==celltype].set_index('mouse')['Neighbor celltype fraction']
        sr = sr1 - sr0
        stat_result = scipy.stats.kstest(sr0, sr1,)
        print(f'KS test {comparison} vs', celltype, ':\tp=%.3g' %stat_result.pvalue)

# Single cell correlation plots
def corr_df1(adata, celltype_pairs, layer='dcnv_0.25sigma'):
    # Build Xcorr dataframe, replace diagonal with nans
    Xcorr = adata.varp[f'corr_{layer}'].copy()
    np.fill_diagonal(Xcorr, np.nan)
    Xcorr = pd.DataFrame(Xcorr, columns=adata.var_names, index=adata.var_names)
    
    rows = []
    for celltype_pair in celltype_pairs:
        var_names = [adata.var_names[adata.var['celltype'] == celltype] for celltype in celltype_pair]
        corrmean = np.nanmean(Xcorr.loc[var_names[0], var_names[1]])
        row = pd.Series({
                'correlation': corrmean,
              'celltype_pair': str(celltype_pair),
              'region': adata.uns['metadata']['region'],
              'mouse': adata.uns['metadata']['mouse'],})
        rows.append(row)
    df = df.concat(rows, axis=1).T
    return df

def corr_df(adatas, celltype_pairs, region=None, layer='X'):
    df = pd.concat([corr_df1(adata, celltype_pairs, layer=layer) for adata in adatas], axis=0)
    df = df.groupby(['mouse', 'region', 'celltype_pair'], as_index=False)['correlation'].mean()
    if region is not None:
        df = df[df['region']==region]
    celltype_pair_categories = [str(pair) for pair in celltype_pairs]
    df['celltype_pair'] = pd.Categorical(df['celltype_pair'], categories=celltype_pair_categories, ordered=True)
    df['region'] = pd.Categorical(df['region'], categories=['PPC', 'RSC'], ordered=True)
    return df

def plot_celltype_pair_correlations(adatas, layer='dcnv_0.25sigma', region=None, figsize=None, s=5):
    n_mice = len(np.unique([adata.uns['metadata']['mouse'] for adata in adatas]))
    print(f'{n_mice} mice, {len(adatas)} sessions')
    
    celltype_pairs = [('B+R+', 'B+R+'),
                  ('R+', 'R+'),
                  ('Negative', 'Negative'),
                  ('B+R+', 'Negative'),
                  ('B+R+', 'R+'),
                  ('R+', 'Negative')]
    df = corr_df(adatas, celltype_pairs, region=region, layer=layer)
    stats_test_correlations(df, celltype_pairs, comparison=('B+R+', 'B+R+'))
    
    plt.figure(figsize=figsize)
    plt.suptitle(region)
    ax = plt.subplot()
    chart = sns.barplot(data=df, x='celltype_pair', y='correlation', color='grey', errwidth=0.25, zorder=1)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
    sns.swarmplot(data=df, x='celltype_pair', y='correlation', color='black', zorder=2, s=s)
    format_plot(ax)
    ax.tick_params(axis='x', which='both', length=0)
    
def stats_test_correlations(df, celltype_pairs, comparison):
    comparison = str(comparison)
    sr0 = df[df['celltype_pair']== comparison].set_index('mouse')['correlation'] 
    # print(comparison, scipy.stats.kstest(sr0, 'norm'))
    for celltype_pair in celltype_pairs:
        celltype_pair = str(celltype_pair)
        if celltype_pair == comparison: continue
        sr1 = df[df['celltype_pair']==celltype_pair].set_index('mouse')['correlation']
        stat_result = scipy.stats.kstest(sr0, sr1,)
        print(f'KS test {comparison} v', celltype_pair, ':\tp=%.3g' %stat_result.pvalue)

# Compose plots
def obs_keys_set_to_obs_cols(obs_keys_set):
    obs_cols = []
    for obs_keys in obs_keys_set:
        for obs_key in obs_keys:
            obs_cols.extend(obs_key.keys())
    obs_cols = set(obs_cols)
    return obs_cols

def _trigger_data(datas, trig_kwargs, obs_keys, plot_kwargs):
    layers = [d['col'] for d in plot_kwargs]
    if type(obs_keys[0]) is not list:
        obs_keys = [obs_keys]
    obs_cols = obs_keys_set_to_obs_cols(obs_keys)
    
    if type(datas) is not list: 
        datas = [datas]
    
    if type(datas[0]) is not list:
        if type(datas[0]) is AnnData:
            trig = an.trigger_adatas(datas, layers=layers, obs_cols=obs_cols, **trig_kwargs)
        elif type(datas[0]) is pd.core.frame.DataFrame:
            trig = an.trigger_dfs(datas, layers=layers, obs_cols=obs_cols, **trig_kwargs)
    else:
        assert obs_cols == set()
        mice = datas # datas is a list of lists of adatas
        print('here')
        trig = an.trigger_adatas_mean(mice, layers=layers, obs_cols=obs_cols, **trig_kwargs)
    return trig

def subgrid_trig_1d(data, trig=None, gsi=None, trig_kwargs={}, obs_keys_set=None, 
                    plot_kwargs=None, show_ci=True, overlay_kwargs=[{}], flip=None, vlines=[0], subsample=False, trial_range=None):
    if obs_keys_set is None:
        obs_keys_set = [[{}]]

    if trig is None:
        trig = _trigger_data(data, trig_kwargs, obs_keys_set, plot_kwargs)

    if trial_range is not None:
        trig = trig[slice(*trial_range)]
        
    t = trig.var['t']
    tlim = (t.iloc[0], t.iloc[-1])

    # Define gridspec
    if gsi is None:
        gs0 = mpl.gridspec.GridSpec(nrows=1, ncols=1)
        gsi = gs0[0, 0]
    gsi_parent = gsi
    if flip:
        assert len(obs_keys_set) == 2
        ncols = 1
    else:
        ncols = len(obs_keys_set)
    gs = gsi_parent.subgridspec(nrows=len(plot_kwargs), ncols=ncols, hspace=0.2)

    def format_axis(ax):
        ax.tick_params(axis='both', which='both', width=0.25)
        if j > 0 and (flip is None):
            # ax.yaxis.set_ticklabels([])
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_visible(False)
        if yticks is not None:
            ax.set_yticks(yticks)

        # Annotation
        if hline is not None:
            ax.axhline(hline, c='grey', lw=0.25, ls='--')
        if i < len(plot_kwargs)-1:
            for spine in ['top', 'right', 'bottom']:
                ax.spines[spine].set_visible(False)
                ax.xaxis.set_visible(False)
                ax.set_xticks([])
        else:
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.set_xlabel('Time (s)')
        add_vlines(ax, vlines, c='grey', ls='--', lw=0.25)
        if j == 0 or flip is not None:
            ax.set_ylabel(col, rotation=0, ha='right')
        ax.set_ylim(ylim)
        ax.set_xlim(tlim)

    # Loop through plots
    d = {}
    for i, kwargs in enumerate(copy.deepcopy(plot_kwargs)):
        # Extract arguments
        col = kwargs.pop('col')
        statistic = kwargs.pop('statistic', np.nanmean)
        hline = kwargs.pop('hline', None)
        ylim = kwargs.pop('ylim', None)
        yticks = kwargs.pop('yticks', None)
        color = kwargs.pop('color', 'black')
        
        data = {}
        for j, obs_keys in enumerate(obs_keys_set):
            data[j] = {}
            # Loop through lines
            for k, obs_key in enumerate(obs_keys):
                if i == 0:
                    kwargs.update(dict(label=str(obs_key), legend_loc=(0, 1.1)))
                idy = an.fetch_index(trig.obs, obs_key)
                data[j][k] = trig[idy].layers[col]
                if flip is None:
                    if k == 0:
                        ax = plt.subplot(gs[i, j])
                    kwargs.update(dict(color=color))
                    kwargs.update(overlay_kwargs[k])
                    if i==0:
                        print(f'{str(overlay_kwargs[k])}: {len(data[j][k])} trials')
                    lineplot(data[j][k], xticks=t, statistic=statistic, show_ci=show_ci, **kwargs)
                    
                    format_axis(ax)
                    d[col] = data
        
        if flip is not None:
            ax = plt.subplot(gs[i, 0])
            d[col] = {}
            d[col][0] = {}
            for k in range(len(obs_keys)):
                if col in flip:
                    datajk = np.concatenate([data[0][k], data[1][k]*-1], axis=0)
                else:
                    datajk = np.concatenate([data[0][k], data[1][k]], axis=0)
                d[col][0][k] = datajk
                
                if i==0:
                    print(f'{str(overlay_kwargs[k])}: {len(datajk)} trials')

                kwargs.update(dict(color=color))
                kwargs.update(overlay_kwargs[k])
                lineplot(datajk, xticks=t, statistic=statistic, show_ci=show_ci, **kwargs)
            format_axis(ax)
            
    d['t'] = t
    return d

def subgrid_trig_2d(adatas=None, trig=None, gsi=None, trig_kwargs={}, obs_keys=None, plot_kwargs=None, palette=None, vlines=[0], trial_range=None):
    # Compute triggered data
    if obs_keys is None:
        obs_keys = [{}]
    
    if trig is None:
        trig = _trigger_adatas(adatas, trig_kwargs, obs_keys, plot_kwargs)

    if trial_range is not None:
        trig = trig[slice(*trial_range)]

    # Define gridspec
    if gsi is None:
        gs0 = mpl.gridspec.GridSpec(nrows=1, ncols=1)
        gsi = gs0[0, 0]
    gsi_parent = gsi
    gs = gsi_parent.subgridspec(nrows=len(obs_keys), ncols=len(plot_kwargs), hspace=0.2, wspace=0.6)
    
    # Loop through plots
    t = trig.var['t']
    tlim = (t[0], t[-1])
    plot_kwargs = copy.deepcopy(plot_kwargs)
    for j, kwargs in enumerate(plot_kwargs):
        # Sort arguments
        col = kwargs.pop('col')

        for i, obs_key in enumerate(obs_keys):
            ax = plt.subplot(gs[i, j])
            idy = an.fetch_index(trig.obs, obs_key)
            vlim = kwargs.pop('vlim', None)
            if vlim is not None:
                kwargs['vmin'] = vlim[0]
                kwargs['vmax'] = vlim[1]
            pcolormesh(trig[idy].layers[col], xticks=mesh_ticks(t), cbar=False, **kwargs)
            ax.invert_yaxis()
            ax.tick_params(axis='both', which='both', width=0.25)

            if j == 0:
                for spine in ['top', 'right',]:
                    ax.spines[spine].set_visible(False)
            else:
                for spine in ['top', 'right', 'left',]:
                    ax.spines[spine].set_visible(False)
                    ax.yaxis.set_visible(False)
                    ax.set_yticks([])

            # Annotation
            if vlines is not None:
                for vline in vlines:
                    ax.axvline(vline, c='red', lw=0.5, ls='--')
            if i == len(obs_keys)-1:
                ax.set_xlabel('Time (s)')
            else:
                ax.xaxis.set_visible(False)
            if i == 0:
                ax.set_title(col)
            if j == 0:
                ax.set_ylabel(obs_key, rotation=0, ha='right')

# Generic Plot 1d
def plot_trigger_1d(adatas, trigger_name=None, plot_kwargs=None, obs_keys_set=None, min_accuracy=None, max_accuracy=None,
                    figsize=(1, 6), offset_s=0, min_dist_s=5, t_range=(-5, 5), suptitle_y=1.05, **kwargs):
    
    # Select sessions
    if min_accuracy is not None:
        adatas = [adata for adata in adatas if adata.uns['trials']['correct'].mean() >= min_accuracy]
    if max_accuracy is not None:
        adatas = [adata for adata in adatas if adata.uns['trials']['correct'].mean() <= max_accuracy]
    if type(adatas) is not list:
        adatas = [adatas]
        
    # Print session stats
    n_mice = len(np.unique([adata.uns['metadata']['mouse'] for adata in adatas]))
    print(f'{n_mice} mice, {len(adatas)} sessions')
    
    # Plot
    plt.figure(figsize=figsize)
    plt.suptitle(sessions_title(adatas), y=suptitle_y)
    trig_kwargs = dict(trigger_name=trigger_name, min_dist_s=min_dist_s, t_range=t_range, offset_s=offset_s)
    d = subgrid_trig_1d(adatas, trig_kwargs=trig_kwargs, obs_keys_set=obs_keys_set, plot_kwargs=plot_kwargs, **kwargs)
    # trig1d_stats(d, tlim=(0.5, 2.5), tlim0=(-1, 0))

    return d

def trig1d_stats_within_row(d, tlim, tlim0=(-1, 0), rows=None, conditions=[0, 1], col=0):
    print(f'Stats {tlim[0]} to {tlim[1]} s minus {tlim0[0]} to {tlim0[1]} s.')
    t = d['t']
    idt = (t >= tlim[0]) & (t < tlim[1])
    idt0 = (t >= tlim0[0]) & (t < tlim0[1])
    
    for row in rows:
        x, y = [d[row][col][i][:, idt].mean(axis=1)-d[row][col][i][:, idt0].mean(axis=1) for i in conditions]
        stat_result = scipy.stats.ranksums(x, y)
        print('Wilcoxon rank-sum', row, f', condition {conditions[0]} v {conditions[1]}:', ',\tp=%.3g' %stat_result.pvalue)

def trig1d_stats_across_rows(d, tlim, tlim0=(-1, 0), row_pairs=None, condition=0, col=0):
    print(f'Stats {tlim[0]} to {tlim[1]} s minus {tlim0[0]} to {tlim0[1]} s.')
    t = d['t']
    idt = (t >= tlim[0]) & (t < tlim[1])
    idt0 = (t >= tlim0[0]) & (t < tlim0[1])
    
    for row_pair in row_pairs:
        i = condition
        x, y = [d[row][col][i][:, idt].mean(axis=1)-d[row][col][i][:, idt0].mean(axis=1) for row in row_pair]
        stat_result = scipy.stats.wilcoxon(x, y)
        print('Paired Wilcoxon %s v %s' %(row_pair[0], row_pair[1]), f', condition {condition}: \tp=%.3g' %stat_result.pvalue)

def trig1d_stats_across_ds(d1, d2, tlim, tlim0, row, condition=0, col=0):
    print(f'Stats {tlim[0]} to {tlim[1]} s minus {tlim0[0]} to {tlim0[1]} s.')
    t = d1['t']
    idt = (t >= tlim[0]) & (t < tlim[1])
    idt0 = (t >= tlim0[0]) & (t < tlim0[1])
    
    i = condition
    x, y = [d[row][col][i][:, idt].mean(axis=1)-d[row][col][i][:, idt0].mean(axis=1) for d in [d1, d2]]
    stat_result = scipy.stats.ranksums(x, y)
    print('Wilcoxon rank-sum', row, f', condition {condition}:', ',\tp=%.3g' %stat_result.pvalue)

# Plot trajectories overlayed with activity
def plot_colored_trajectory_trial(adata, trial, x, y, ax, c=None, cmap='viridis', vmin=0, vmax=1):
    idy = (adata.obs['trial']==trial)&(adata.obs['inITI']==False)
    x_vals = adata[idy].obs[x].to_numpy()
    y_vals = adata[idy].obs[y].to_numpy()
    c_vals = adata[idy].obs[c].to_numpy()
    lines = colored_line(x_vals, y_vals, c_vals, cmap=cmap, ax=ax, vmin=vmin, vmax=vmax)

def plot_trajectory_extreme_activity(adata, plot_kws, x='h', y='y', world='white_right', pcntl=0.5, sort_signal='Sst44_dcnv_0.25sigma_norm', max_trial_len=10, sort_y_range=(0, 330), xlim=(-np.pi, np.pi), ylim=(-10, 315)):
    # Setup figure
    worlds = sorted(adata.obs['world'].unique())
    fig, axs = plt.subplots(2, len(plot_kws), figsize=(3*len(plot_kws), 3*2))
    fig.set_tight_layout(True)

    # Figure title
    accuracy = adata.uns['trials']['correct'].mean()
    md = adata.uns['metadata']
    fig.suptitle('%s %s (%.2f accuracy) %i Sst44 cells %s'%(md['mouse'], md['date'], accuracy, (adata.var['celltype']=='B+R+').sum(), world))

    # Plotting function for each batch of trials
    def plot_trials(trials, plot_kw, ax):
        for i, trial in enumerate(trials):
            plot_colored_trajectory_trial(adata, trial, x, y, ax,**plot_kw)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

    def _format_plot(ax, i_plot):
        if i_plot == 0:
            remove_spines=['top', 'right']
        else:
            remove_spines = ['top', 'right', 'left']
        format_plot(ax, remove_spines)
    
    # Get extreme trials
    idy = (adata.obs['world']==world)&(adata.obs['trial_len_s']<max_trial_len)&(adata.obs.inITI==False)&(adata.obs.y>=sort_y_range[0])&(adata.obs.y<sort_y_range[1])
    trial_sorted_activity = adata[idy].obs.groupby('trial')[sort_signal].max().sort_values().index
    trial_lim = int(len(trial_sorted_activity)*pcntl)

    # Plot low activity trials
    for i_plot, plot_kw in enumerate(plot_kws):

        # Plot high activity trials
        ax = axs[0][i_plot]
        ax.set_title(plot_kw['c'])
        if i_plot == 0:
            ax.set_ylabel(f'lowest activity {world}\n{y}', rotation=90)
        plot_trials(trial_sorted_activity[:trial_lim], plot_kw, ax)
        _format_plot(ax, i_plot)

        # Plot high activity trials
        ax = axs[1][i_plot]
        if i_plot == 0:
            ax.set_ylabel(f'highest activity {world}\n{y}')
        plot_trials(trial_sorted_activity[-trial_lim:], plot_kw, ax)
        _format_plot(ax, i_plot)

# Plot binned activity
def plot_Xbin1_cellmean(adatas, obs=None, bins=None, obs_key=None, var_key=None, norm=False, color='black', **pcolormesh_kwargs):
    means = [an.binX1(adata, obs=obs, bins=bins, obs_key=obs_key, var_key=var_key, norm=norm).mean(axis=0) for adata in adatas]
    mean = np.nanmean(means, axis=0)
    plt.plot(bins[:-1], mean, label=str(var_key))
    plt.legend()

def plot_Xbin1(adatas, obs=None, bins=None, obs_key=None, var_key=None, norm=False, sort=False, **pcolormesh_kwargs):
    if type(adatas) is not list:
        adatas = [adatas]
    dfs = [an.binX1(adata, obs=obs, bins=bins, obs_key=obs_key, var_key=var_key, norm=norm) for adata in adatas]
    df = pd.concat(dfs)
    if sort:
        cell_order = df.idxmax(axis=1).sort_values().index
        df = df.loc[cell_order]
    pcolormesh(np.array(df), xticks=bins, **pcolormesh_kwargs)
    # ax = sns.heatmap(data=df, vmin=vmin, vmax=vmax, xticklabels=xticklabels, yticklabels=False, cmap='viridis')

def plot_binned_statistic_2d(adatas, x, y, values, xlim=None, ylim=None, xticks=None, yticks=None, bins=30, vmin=0, vmax=0.5, min_accuracy=0, max_trial_len_s=np.inf, obs_key=None, min_stderr=0.5, figsize=(4, 4), lw=0.5, frame_shift=0):
    # Compile data across sessions, filter
    adatas = [adata for adata in adatas if adata.uns['trials']['correct'].mean()>=min_accuracy]
    obs_cat = pd.concat([adata.obs for adata in adatas], axis=0)
    obs_cat = obs_cat[~obs_cat['inITI']]
    obs_cat = obs_cat[obs_cat['trial_len_s']<max_trial_len_s]
    obs_cat = obs_cat[~obs_cat[[x, y, values]].isna().any(axis=1)]
    obs_cat = obs_cat[an.fetch_index(obs_cat, obs_key)]
    
    def bootstrap_stderr(arr1d,):
        assert len(arr1d.shape)==1
        arr1d = arr1d[~np.isnan(arr1d)]
        
        if len(arr1d) < 3:
            return np.inf

        if np.std(arr1d) == 0:
            return 0

        bootstrap_result = scipy.stats.bootstrap((arr1d,), statistic=np.nanmean, confidence_level=0.95, n_resamples=20,)
        return bootstrap_result.standard_error
    
    # Compute binned statistic
    mean, x_edge, y_edge, binnumber = scipy.stats.binned_statistic_2d(x=obs_cat[x], y=obs_cat[y], values=obs_cat[values], statistic=np.nanmean, bins=bins, range=[xlim, ylim], expand_binnumbers=False)
    
    # Mask bins where stderr > range * min_stderr
    if min_stderr is not None:
        stderr, _, _, _ = scipy.stats.binned_statistic_2d(x=obs_cat[x], y=obs_cat[y], values=obs_cat[values], statistic=bootstrap_stderr, bins=bins, range=[xlim, ylim], expand_binnumbers=False)
        mean[stderr > (mean*min_stderr)] = np.nan
    
    # Plot
    # plt.figure(figsize=figsize)
    # mouse_ids = sorted(set([adata.uns['metadata']['mouse'] for adata in adatas]))
    # plt.suptitle(', '.join(mouse_ids)) 
    # plt.suptitle(values)   
    viridis = copy.copy(mpl.cm.viridis)
    viridis.set_bad('darkgrey')
    plt.pcolormesh(x_edge, y_edge, mean.T, cmap=viridis, vmin=vmin, vmax=vmax)
    plt.xlabel(x)
    plt.ylabel(y)
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', width=lw)

    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)

    cbar = plt.colorbar(aspect=5, shrink=0.5)
    cbar.ax.tick_params(axis='both', which='both', width=lw)
    # cbar.ax.set_title(values)
    
def bootstrap_ci(x, confidence_level=0.95, n_resamples=10000):
    result = scipy.stats.bootstrap((x,), statistic=np.nanmean, confidence_level=confidence_level, n_resamples=n_resamples)
    return result.confidence_interval

def plot_bin1d(df, x, y, color='black', label=None, bins=None, qbins=None, cut=None, show_confidence_interval=True):
    # Bin x values
    if cut is None:
        if qbins is not None:
            cut = pd.qcut(df[x], q=qbins)
        else:
            cut = pd.cut(df[x], bins=bins)
    grouped = df[y].groupby(cut)
    
    # Plot
    yi = grouped.mean()
    xi = yi.index.dtype.categories.mid
    plt.plot(xi, yi, color=color, label=label, lw=0.5)
    plt.scatter(xi, yi, color=color, s=2)
    if show_confidence_interval:
        bootstrap = grouped.apply(bootstrap_ci)
        yerr = np.abs(np.array([[r.low, r.high] for r in bootstrap]).T - np.array(yi))
        plt.errorbar(xi, yi, yerr=yerr, color=color, lw=0.25)
    plt.xticks(bins)
    plt.xlabel(x)
    plt.ylabel(y)
    ax = plt.gca()
    ax.axhline(0, c='black', lw=0.25, ls='--')
    format_plot(ax)

    return grouped, cut

# Plotting primitives
def plot_Xtrig2d(adatas, gsi=None, mean_axis=0, key={}, percentile=None, **plot_kwargs):
    # Check inputs
    assert mean_axis in [0, 1]
    
    # Compile Xtrig across all adatas
    Xtrigs = [an.compute_Xtrig(adata, trig=adata.uns['trig']['obs'], **key) for adata in adatas]
    
    # Take mean across mean_axis.
    Xtrigs_mean = [np.nanmean(Xtrig, axis=mean_axis) for Xtrig in Xtrigs]

    # Concatenate and take mean
    Xtrig_cat = np.concatenate(Xtrigs_mean, axis=0)
    
    # If take mean across axis 0, rank trials
    rank_trials = all([adata.uns['trig']['rank_val'] is not None for adata in adatas])
    if mean_axis == 0 and rank_trials:
        rank_vals = compile_rank_vals(adatas, key['obs_key'], trig_df='obs')
        Xtrig_cat = Xtrig_cat[np.argsort(rank_vals)[::-1]]

        if percentile is not None:
            slice_inds = [int(len(Xtrig_cat) * percentile_i / 100) for percentile_i in percentile]
            Xtrig_cat = Xtrig_cat[slice_inds[0]:slice_inds[1]]
    
    # Plot
    if gsi is None:
        ax = plt.subplot()
    else:
        ax = plt.subplot(gsi)
    t = adatas[0].uns['trig']['obs']['t']
    pcolormesh(Xtrig_cat, index=t, **plot_kwargs)
    ax.invert_yaxis()
    return ax

def plot_bin_ITItrig(adata, key, obs_bin='y', bins=np.arange(0, 320, 10), celltypes=None, height_ratios=None, parent_gridspec=None):
    adata = adata.copy()
    notITI = an.fetch_index(adata.obs, {'inITI': 0})
    nrows = len(celltypes)
    ntrials = get_key_ntrials(adata, key)
    if parent_gridspec is None:
        plt.figure(figsize=(8, 6))
        title = '\n'.join([session_title(adata), key_title(key), f'{ntrials} trials'])
        plt.suptitle(title, y=1.1)
        gs = mpl.gridspec.GridSpec(nrows=nrows, ncols=2, height_ratios=height_ratios, width_ratios=[1, 1])
    else:
        gs = parent_gridspec.subgridspec(nrows=nrows, ncols=2, height_ratios=height_ratios, width_ratios=[1, 1])
    
    plot_kwargs = dict(norm=True, vmin=0, vmax=1.5, cbar=False)    
    for irow, celltype in enumerate(celltypes):
        var_key = {'celltype': celltype}
        if irow < nrows - 1:
            tick_params = dict(left=False, bottom=False, labelleft=False, labelbottom=False)
            set_xlabel = False
        else:
            tick_params = dict(left=False, bottom=True, labelleft=False)
            set_xlabel = True
        
        if key['correct']:
            vlines = [0, 3]
        else:
            vlines = [0, 7]
            
        plot_kwargs.update(dict(var_key=var_key))
        
        ax = plt.subplot(gs[irow, 0])
        if irow == 0:
            _key_title = key_title(key).replace('\n', ', ')
            title = f'{_key_title}, {ntrials} trials \n{celltype}'
            ax.set_title(title, loc='left')
        else:
            ax.set_title(celltype, loc='left')
        plot_Xbin1(adata[notITI], obs=obs_bin, bins=bins, obs_key=key, **plot_kwargs)
        ax.tick_params(**tick_params)
        if set_xlabel: ax.set_xlabel(obs_bin)
        
        ax = plt.subplot(gs[irow, 1])            
        plot_Xtrig(adata, trigger='inITI', t_range=(-1, 8), obs_key=key, **plot_kwargs)
        ax.tick_params(**tick_params)
        for vline in vlines: ax.axvline(vline, color='white', ls='--')
        if set_xlabel: ax.set_xlabel('Time (s)')

# Scrolling plots for raw data visualization
def draw_tmaze(ax, color='lightgrey'):
    arm_top = 307.6
    arm_bot = 305.4
    arm_left = -50
    arm_right= 50
    funnel_left = -1.1
    funnel_right = 1.1
    stem_left = -0.2
    stem_right = +0.2
    stem_bot = 0
    stem_top = 300
    funnel_bot = 301
    reward_left = -12
    reward_right = 12
    lines = [([arm_left, arm_right], [arm_top, arm_top]),
             ([arm_left, funnel_left], [arm_bot, arm_bot]),
             ([funnel_right, arm_right], [arm_bot, arm_bot]),
             ([arm_left, arm_left], [arm_bot, arm_top]),
             ([arm_right, arm_right], [arm_bot, arm_top]),
             ([stem_left, stem_left], [stem_bot, stem_top]),
             ([stem_left, funnel_left], [stem_top, funnel_bot]),
             ([funnel_left, funnel_left], [funnel_bot, arm_bot]),
             ([stem_right, stem_right], [stem_bot, stem_top]),
             ([stem_right, funnel_right], [stem_top, funnel_bot]),
             ([funnel_right, funnel_right], [funnel_bot, arm_bot])]
    dashed_lines = [
            ([reward_left, reward_left], [arm_bot, arm_top]),
            ([reward_right, reward_right], [arm_bot, arm_top])]
    
    for line in lines:
        ax.plot(*line, c=color, lw=1, zorder=1)

    for line in dashed_lines:
        ax.plot(*line, c=color, lw=1, zorder=1, ls='--')

def draw_lineartrack(ax, color='lightgrey'):
    stem_left = -10
    stem_right = +10
    stem_bot = 0
    stem_top = 220
    lines = [([stem_left, stem_left], [stem_bot, stem_top]),
             ([stem_right, stem_right], [stem_bot, stem_top]),]

    for line in lines:
        ax.plot(*line, c=color, lw=1, zorder=1)

    reward = (0, 200)
    ax.scatter(*reward, c='tab:blue', s=30)

def quiver_trajectory(vr, quiver=True, angle='h', show_ITI=True, linecolor='t', cmap=None, show_start=True, **kwargs):
    if type(angle) is str:
        angle = (vr[angle] * -1) + (np.pi / 2)
    dx = np.cos(angle)
    dy = np.sin(angle)
    x, y = vr['x'], vr['y']
    c = vr[linecolor]
    inITI = vr['inITI']
    x_notITI = np.ma.masked_where(inITI, x)
    y_notITI = np.ma.masked_where(inITI, y)
    c_notITI = np.ma.masked_where(inITI, c)
    dx_notITI = np.ma.masked_where(inITI, dx)
    dy_notITI = np.ma.masked_where(inITI, dy)
    
    plt.scatter(x.iloc[0], y.iloc[0], color='red', s=1, zorder=10)
    if cmap is not None:
        colored_line(x_notITI, y_notITI, c=c_notITI, cmap=cmap)
    else:
        plt.plot(x_notITI, y_notITI, c='black', lw=0.25, zorder=2)

    if show_ITI:
        x_inITI = np.ma.masked_where(inITI==False, x)
        y_inITI = np.ma.masked_where(inITI==False, y)
        plt.plot(x_inITI, y_inITI, c='red', zorder=2)
    # lick_colormap = ['black', 'red']
    if quiver:
        if show_ITI:
            xi, yi, dxi, dyi = x_notITI, y_notITI, dx_notITI, dy_notITI
        else:
            xi, yi, dxi, dyi = x, y, dx, dy

        default_kwargs = dict(scale=60, pivot='middle', zorder=3)#, cmap='copper',)
        default_kwargs.update(kwargs)

        plt.quiver(xi, yi, dxi, dyi, vr['lick'].astype(int), **default_kwargs)
        ax = plt.gca()
        ax.set_aspect('equal')
     
def quiver_activity(adata, obs_col, ax=None, scale=30, show_ITI=True, cmap='viridis', **kwargs):
    if ax is None:
        ax = plt.gca()
    x = adata.obs['x']
    y = adata.obs['y']
    angle = (adata.obs['h'] * -1 + np.pi / 2)
    dx = np.cos(angle)
    dy = np.sin(angle)

    if not show_ITI:
        inITI = adata.obs['inITI']
        x = np.ma.masked_where(inITI, x)
        y = np.ma.masked_where(inITI, y)
        dx = np.ma.masked_where(inITI, dx)
        dy = np.ma.masked_where(inITI, dy)
    # activity = adata.X.mean(axis=1)
    activity = adata.obs[obs_col]
    quiver = ax.quiver(x, y, dx, dy, activity, scale=scale, pivot='middle', cmap=cmap, zorder=4, **kwargs)
    return quiver

def plot_trajectory(adata, tlim=None, trial=None, vmax=0.8, xlim=None, ylim=None, equal_aspect=False, step=1, cmap='viridis', scale=7):
    t = adata.obs['t']
    if trial is None:
        assert tlim is not None
        adatai = adata[(t>=tlim[0]) & (t<tlim[1])]
    else:
        adatai = adata[adata.obs['trial']==trial]
    adatai = adatai[adatai.obs['inITI']==False]
    if len(adatai) == 0:
        return
    tstart = adatai.obs['t'].iloc[0]
    tend = adatai.obs['t'].iloc[-1]
    
    plt.figure(figsize=(0.9, 0.9))
    ax = plt.subplot()
    q = quiver_activity(adatai[::step], obs_col='Sst44_dcnv_norm', scale=scale, show_ITI=False, width=0.05, cmap=cmap)
    # if (tend >= tlim[0]) & (tend < tlim[1]):
    #     ind = (adatai.obs['t'] >= tend)
    #     quiver_activity(adatai[ind], obs_col='Sst44_dcnv_norm', scale=7, show_ITI=True, width=0.05)
    # if (tstart >= tlim[0]) & (tend < tlim[1]):
    #     quiver_activity(adatai[0], obs_col='Sst44_dcnv_norm', scale=7, show_ITI=True, width=0.05)
    q.set_clim(0, vmax)

    # draw_maze(ax)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    
def plot_trajectory_animation(adata, trange, tstep=0.6, save=True, xlim=None, ylim=None, **kwargs):
    mouse = adata.uns['metadata']['mouse']
    date = adata.uns['metadata']['date']
    trial = adata[np.where(adata.obs['t']>=trange[0])[0][0]].obs['trial'][0]
    print(f'Trial {trial}')

    # Define spatial range
    if xlim is None:
        xlim = [adata.obs['x'].min(), adata.obs['x'].max()]
        xrange = xlim[1] - xlim[0]
        if xrange < 2:
            xlim = [xlim[0]-1, xlim[1]+1]
            xrange = xlim[1] - xlim[0]
        xlim = [xlim[0]-xrange*0.05, xlim[1]+xrange*0.05]
    if ylim is None:
        ylim = [adata.obs['y'].min(), adata.obs['y'].max()]
        yrange = ylim[1] - xlim[0]
        ylim = [ylim[0]-yrange*0.05, ylim[1]+yrange*0.05]
    
    # Plot
    for tstart in np.arange(*trange, tstep):
        tlim = np.array([tstart, tstart+tstep])
        print(tlim)
        plot_trajectory(adata, tlim, xlim=xlim, ylim=ylim, **kwargs)
        if save:
            plt.savefig(f'{mouse}_{date}_trajectory_trial{trial}_%.1fs-%.1fs.pdf' %(tstart, tstart+tstep))

def plot_trajectory_animation_trial(adata, trial, tstep=0.6, save=True, **kwargs):
    # Select trial
    adatai = adata[(adata.obs['trial']==trial) & (adata.obs['inITI']==False)].copy()
    adatai.obs['t'] -= adatai.obs['t'].iloc[0]

    # Define trange
    trange = (adatai.obs['t'].iloc[0], adatai.obs['t'].iloc[-1])
    plot_trajectory_animation(adatai, trange, tstep=tstep, save=save, **kwargs)

def plot_trajectory_animation_peak_activity(adata, trial, t_window=(-2, 2), tstep=1.3, save=True, **kwargs):
    # Select trial
    adatai = adata[(adata.obs['trial']==trial) & (adata.obs['inITI']==False)].copy()
    adatai.obs['t'] -= adatai.obs['t'].iloc[0]

    # Define trange
    peak_ts = an.activity_peak_ts(adatai, signal='Sst44_dcnv_0.25sigma_norm')
    peak_t = peak_ts[0]
    adatai.obs['t'] -= peak_t
    adatai = adatai[(adatai.obs['t']>=t_window[0]) & (adatai.obs['t']<t_window[1])]
    trange = t_window
    plot_trajectory_animation(adatai, trange, tstep=tstep, save=save, **kwargs)

def plot_median_trajectory(adata, interp_bins=309):
    obs = adata.obs
    traj = an.median_trajectory(obs)
    
    plt.figure(figsize=(15, 5))
    plt.suptitle(session_title(adata))
    
    ax = plt.subplot(131)
    plot_trajectory(traj)
    
    traj = an.fix_trajectory(traj)
    ax = plt.subplot(132)
    plot_trajectory(traj)
    
    traj = an.interpolate_trajectory(traj, bins=interp_bins)
    ax = plt.subplot(133)
    plot_trajectory(traj)

def plot_session(adata, columns=['X'], tlim=None, trial=None, width_ratios=None, hlines=[], hline_color='tab:blue', time_tick_step=5, timescale=0.2, y_title=1.1, figsize=None, gsi=None, vmax=1.0, quiver_step=3, quiver_scale=7, lw=1, show_start=True, show_end=True,):
    adata = adata.copy()

    if trial is not None:
        adata = adata[adata.obs['trial']==trial]
        adata.obs['t'] -= adata.obs['t'].iloc[0]
    
    if tlim is None:
        tlim = tuple(adata.obs['t'].iloc[[0, -1]])
    if tlim[0] < adata.obs.t.iloc[0]:
        tlim = (adata.obs.t.iloc[0], tlim[1])
    if tlim[1] > adata.obs.t.iloc[-1]:
        tlim = (tlim[0], adata.obs.t.iloc[-1])
    
    idt = (adata.obs['t'] >= tlim[0]) & (adata.obs['t'] < tlim[1])
    adata = adata[idt]
    vr = adata.obs
    vr = vr[(vr.t >= tlim[0]) & (vr.t < tlim[1])]
    
    start_inds = fc.rising_idx(adata.obs['inITI'], direction='falling')
    stop_inds = fc.rising_idx(adata.obs['inITI'], direction='rising')
    t_iti_start = adata.obs['t'].iloc[start_inds]
    t_iti_end = adata.obs['t'].iloc[stop_inds]

    def shade_ITI(ax, xlim):
        ITI_start = start_inds.copy()
        ITI_end = stop_inds.copy()
        if ITI_end[0] < ITI_start[0]:
            ITI_start = np.insert(ITI_start, adata.obs['t'].iloc[0], 0)
        if ITI_end[-1] < ITI_start[-1]:
            ITI_end = np.insert(ITI_end, adata.obs['t'].iloc[-1], -1)
        assert len(ITI_start) == len(ITI_end)
        for t0, t1 in zip(ITI_start, ITI_end):
            ax.fill_between(xlim, y1=t0, y2=t1, color='grey')

    if gsi is None:
        if figsize is None:
            figsize = (10, np.ptp(adata.obs['t'])*timescale)
        f = plt.figure(figsize=figsize) #0.4
        plt.suptitle(session_title(adata) + f'\ntlim: {tlim}', y=y_title) #0.885
    gsi = update_gsi(gsi)
    gs = gsi.subgridspec(nrows=1, ncols=len(columns), width_ratios=width_ratios)
    
    for icol, column in enumerate(columns):
        if icol == 0: 
            ax0 = plt.subplot(gs[0, 0])
            yticks = np.arange(np.ceil(tlim[0]), np.floor(tlim[1]), time_tick_step) 
            ax = ax0
            ax.set_yticks(yticks)
        else:
            ax = plt.subplot(gs[0, icol], sharey=ax0)
            ax.yaxis.set_visible(False)
        ax.tick_params(width=lw)
        
        if column['signal'] in adata.layers.keys():
            key = column.copy()
            hue = key.pop('signal')
            var_key = key.pop('var_key', {})
            var_idx = an.fetch_index(adata.var, var_key)
            adatai = adata[:, var_idx].copy()
            xticks = mesh_ticks(range(adatai.n_vars))
            yticks = mesh_ticks(adata.obs['t'])
            
            if hue == 'X':
                X = adatai.X
            else:
                X = adatai.layers[hue]
            ax.pcolormesh(xticks, yticks, X, zorder=1, vmax=vmax, lw=lw,)
            ax.set_ylabel('Time (s)')
            ax.set_xlabel(str(var_key))
            ax.xaxis.set_visible(False)
        elif column['signal'] == 'wall_color':
            # values = sorted(adata.obs[column].unique())
            # mapping = {val: i for val, i in zip(values, range(len(values)))}
            mapping = {'white': 0, 'black': 1}
            numerical = np.array(adata.obs[column['signal']].map(mapping).astype(int))
            xticks = np.arange(2)
            yticks = mesh_ticks(adata.obs['t'])
            ax.pcolormesh(xticks, yticks, numerical[:, None], cmap='Greys', vmin=0, vmax=1)
            ax.xaxis.set_visible(False)
        elif column['signal'] == 'reward':
            print(column['signal'])
            ax.plot(np.array(vr[column['signal']]), np.array(vr['t']), c='tab:red')
            ax.axis('off')
            ax.set_xlabel(column)
            ax.xaxis.set_visible(False)
        elif column['signal'] == 'lick':
            vals = np.array(vr[column['signal']])
            xticks = np.arange(2)
            yticks = mesh_ticks(np.array(vr['t']))
            ax.pcolormesh(xticks, yticks, vals[:, None], cmap='Blues')
            ax.set_xlabel(column)
            ax.xaxis.set_visible(False)
        elif column['signal'] in vr.columns:
            kwargs = column.copy()
            signal = kwargs.pop('signal',)
            xticks = kwargs.pop('xticks', None)
            xlim = kwargs.pop('xlim', None)
            # shade_ITI(ax, xlim)
            if pd.api.types.is_numeric_dtype(vr[signal]):
                ax.plot(vr[signal], vr['t'], **kwargs)
            else:
                values = sorted(vr[signal].unique())
                mapping = {val: i for val, i in zip(values, range(len(values)))}
                numerical = vr[signal].map(mapping).astype(int)
                ax.plot(numerical, vr['t'], **kwargs)
            ax.set_xlabel(column['signal'])
            ax.axvline(0, lw=0.25, ls='--', c='grey')
            ax.set_xlim(xlim)
            ax.set_xticks(xticks)
            
            if column == 'dh_error':
                ax.set_xlim(-0.05, 0.05)
            elif column in ['h', 'h_error']:
                ax.set_xlim(-np.pi, np.pi)
        elif column['signal'] == 'quiver':
            kwargs = column.copy()
            lines = kwargs.pop('lines', [-12, 12])
            x = np.array(adata.obs['x'])
            angle = np.array(adata.obs['h'] * -1 + np.pi / 2)
            dx = np.cos(angle)
            dy = np.sin(angle)
            xlim = (-15, 15)
            # shade_ITI(ax, xlim)
            ax.quiver(x[::quiver_step], np.array(adata.obs['t'])[::quiver_step], dx[::quiver_step], dy[::quiver_step], color='black', pivot='middle', scale=quiver_scale, width=0.005, headwidth=9, headlength=15, headaxislength=10)
            ax.set_xlabel(column['signal'])
            xmax = np.abs(x).max()
            ax.set_xlim(xlim)
            for line in lines:
                ax.axvline(line, lw=0.25, ls='--', color='grey')

            ax.set_xticks([-12, 0, 12])
            
        
        ax.set_ylim(tlim)
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='both', width=0.25)
        if show_end:
            add_hlines(ax, t_iti_end, c='grey', ls='--', lw=lw, zorder=3)
        if show_start:
            add_hlines(ax, t_iti_start, c='limegreen', ls='--', lw=lw, zorder=3)
        
        if column['signal'] != 'wall_color':
            for spine in ['bottom', 'left', 'right']:
                ax.spines[spine].set_visible(False)

        if hlines:
            for hline in hlines:
                ax.axhline(hline, lw=0.25, ls='--', c=hline_color)

# Basic quality checks
def plot_scan_trigger_check(s, xlim=(-0.5, 0.5)):
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8, 3))
    scan_idx = fc.rising_idx(s.sync.raw['ScanImage']>0.5)
    
    ts = [s.sync.raw['t'].values[scan_idx[i]] for i in [0, -1]]
    for i, t in enumerate(ts):
        ax = axs[i]
        ax.plot(s.sync.raw['t'].values[::1], s.sync.raw['ScanImage'].values[::1]/5, c='grey', zorder=1, label='ScanImage trigger')
        ax.plot(s.sync.raw['t'].values[::1], s.sync.raw['Pockels'].values[::1], c='black', zorder=1, label='Pockels')
        ax.scatter(s.sync.scan['t'], np.ones(len(s.sync.scan)), c='red', s=10, zorder=2, label='Extracted time points')
        ax.set_xlim(t+xlim[0], t+xlim[1])
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        if i == 1:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_visible(False)
            print('hello')
            plt.legend()
 
# Plotting behavior
def lineplot_obs(obs, cols=[], tlim=None):
    t = obs['t']
    if tlim is not None:
        idt = (t >= tlim[0]) & (t < tlim[1])
    else:
        idt = np.ones(len(obs)).astype(bool)
    
    nplots = len(cols)
    plt.figure(figsize=(10, len(cols)*2))
    gs = mpl.gridspec.GridSpec(nrows=len(cols), ncols=1)
    for i, col in enumerate(cols):
        if i == 0:
            ax0 = ax = plt.subplot(gs[i, 0])
        else:
            ax = plt.subplot(gs[i, 0], sharex=ax0)
        plt.plot(t[idt], obs.loc[idt, col])   
        ax.axhline(0, ls='--', lw=0.25, color='grey')
        ax.set_ylabel(col)

# Plotting sources
def select_source_ids_from_plane(var_names, plane):
    sources = [unpack_var_name(var_name)[1] for var_name in var_names if unpack_var_name(var_name)[0] ==  plane]
    return sources

def plot_cell_footprints_plane(adata, plane, **kwargs):
    sources = select_source_ids_from_plane(adata.var_names, plane)
    stat = np.load(adata.uns['path']['stat_npy'].format(plane=plane), allow_pickle=True)
    md = adata.uns['metadata']
    img = np.zeros((md['Ly'], md['Lx']))
    for stati in stat[sources]:
        img[stati['ypix'], stati['xpix']] = 1
    plt.pcolormesh(img, **kwargs)
    
def plot_cell_footprints(adata, var_key=None, **kwargs):
    var_idx = an.fetch_index(adata.var, var_key)
    adata = adata[:, var_idx]
    nplanes = adata.uns['metadata']['nslices']
    planes = range(1, nplanes+1)
    plt.figure(figsize=(6*nplanes, 6))
    plt.suptitle(session_title(adata), y=1.3)
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=nplanes)
    for i, plane in enumerate(planes):
        ax = plt.subplot(gs[0, i])
        plot_cell_footprints_plane(adata, plane, **kwargs)
        ax.set_aspect('equal')

# Show images of cropped cells
def get_img(adata, channel, plane=1, filter_key=None):
    if filter_key is None:
        filter_key = adata.uns['ops']['var_filter_key'][channel]

    assert plane != 0 # 1-based
    iplane = plane - 1

    chan_dict = {'B': 0, 'G': 1, 'R': 2}
    ichannel = chan_dict[channel]
    img = imread(adata.uns['path']['meanRef'][filter_key][iplane])[:, :, ichannel]

    if (img == 0).all():
        warnings.warn('Image is blank.')

    return img

def get_cell_img(adata, var_name, channel, crop=(60, 60), filter_key=None):
    plane, source = unpack_var_name(var_name)
    stati = fetch_cell_stat(adata, var_name)
    img = get_img(adata, channel, plane, filter_key=filter_key)
    cell_img = fc.crop_img(img, stati['med'], crop=crop)
    return cell_img

def outline_mask(mask, shell=(1, 0)):
    outer = fc.dilate(mask, dilation=shell[0])
    inner = fc.dilate(mask, dilation=shell[1])
    mask = outer
    mask[inner] = False
    return mask

def get_cell_outline(adata, var_name, crop=(60, 60), shell=(1, 0)):
    Ly, Lx = adata.uns['metadata']['Ly'], adata.uns['metadata']['Lx']
    stati = fetch_cell_stat(adata, var_name)

    mask = np.zeros((adata.uns['metadata']['Ly'], adata.uns['metadata']['Lx'])).astype(bool)
    mask[stati['ypix'], stati['xpix']] = True
    outline = outline_mask(mask)
    cell_outline = fc.crop_img(outline, stati['med'], crop=crop)
    return cell_outline

def plot_cell_img(adata, var_name, channel, vmax=None, crop=(60, 60), filter_key=None, outline=True):
    img = get_cell_img(adata, var_name, channel, crop=crop, filter_key=filter_key)
    plt.imshow(img, vmax=vmax)
    plt.colorbar(ax=plt.gca(), shrink=0.5)
    if outline:
        outline = get_cell_outline(adata, var_name, crop=crop, shell=(1, 0))
        plt.imshow(outline, alpha=0.2)

def imshow_cell(adata, var_name, channels=['G'], vmax=None, crop=(60, 60), filter_key=None, outline=True):
    fig = plt.figure(figsize=(len(channels)*5, 5))
    gs = mpl.gridspec.GridSpec(ncols=len(channels), nrows=1)
    for ichannel, channel in enumerate(channels):
        ax = plt.subplot(gs[0, ichannel])
        vmaxi = vmax[ichannel] if vmax is not None else None
        plot_cell_img(adata, var_name, channel, vmax=vmaxi, crop=crop, filter_key=filter_key, outline=outline)
        ax.set_title(channel)

# Time UMAP
def plot_time_umap(adata, plot_keys=[], ncols=4, fig_scale=5, x='umap_x_dcnv_0.25sigma_norm', y='umap_y_dcnv_0.25sigma_norm', s=1):
    nplots = len(plot_keys)
    nrows = int(np.ceil(nplots / ncols))
    plt.figure(figsize=(ncols*fig_scale, nrows*fig_scale))
    plt.suptitle(session_title(adata))
    gs = mpl.gridspec.GridSpec(nrows=nrows, ncols=ncols)

    for iplot, plot_key in enumerate(plot_keys):
        plot_key = plot_key.copy()
        ax_title = plot_key['hue']
        ax = plt.subplot(gs[int(iplot/ncols), int(iplot%ncols)])
        sns.scatterplot(data=adata.obs, x=x, y=y, s=s, linewidth=0, ax=ax, **plot_key)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title(ax_title)
        ax.set_aspect('equal', 'datalim')
        hue = plot_key['hue']
        if type(hue) is str and hue == 'trial_type_ITI':
            plt.legend(bbox_to_anchor=(-0.7, 1), loc=2, borderaxespad=0.)
            # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

def plot_cell_umap(adata, plot_keys=[], ncols=4, fig_scale=5, x='umap_x_dcnv_0.25sigma_norm', y='umap_y_dcnv_0.25sigma_norm', s=20):
    nplots = len(plot_keys)
    nrows = int(np.ceil(nplots / ncols))
    plt.figure(figsize=(ncols*fig_scale, nrows*fig_scale))
    # plt.suptitle(session_title(adata))
    gs = mpl.gridspec.GridSpec(nrows=nrows, ncols=ncols)
    scatterplot_params = dict(data=adata.var, x=x, y=y, s=s)

    for iplot, plot_key in enumerate(plot_keys):
        irow = int(iplot/ncols)
        icol = int(iplot%ncols)
        ax = plt.subplot(gs[irow, icol])
        sns.scatterplot(**scatterplot_params, **plot_key)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        format_plot(ax, remove_spines=['left', 'top', 'right', 'bottom'])    
        ax.set_aspect('equal', 'datalim')

# Cell UMAP
def plot_clustering_param_search(adata, resolution, ncols=4, figscale=5, **leiden_kwargs):
    X_scaled = an.StandardScaler().fit_transform(adata.X)
    M = an.kneighbors_graph(X_scaled.T, n_neighbors=15, mode='distance', metric='euclidean')
    G, W = an.sparsematrix_to_graph(M)
    
    nrows = int(len(resolution) / ncols)+1
    plt.figure(figsize=(ncols*figscale, nrows*figscale))
    gs = mpl.gridspec.GridSpec(nrows=nrows, ncols=ncols)
    clusters = []
    for i, resolution_i in enumerate(resolution):
        leiden = an.igraph.Graph.community_leiden(G, weights=W, resolution_parameter=resolution_i, **leiden_kwargs)
        clusters.append(leiden)
        print('Finished leiden clustering for resolution %.1f.' %resolution_i)
        ax = plt.subplot(gs[int(i/ncols), int(i%ncols)])
        sns.scatterplot(data=adata.var, x='Xumap_x', y='Xumap_y', hue=leiden.membership, s=4, lw=0, ax=ax, palette='tab20')
        ax.set_aspect('equal', 'datalim')
    return clusters

# General UMAP plot
def plot_umaps(df, plot_dicts, x='Xumap_x', y='Xumap_y', ncols=3, s=10, lw=0):
    nplots = len(plot_dicts)
    nrows = int(np.ceil(nplots/ncols))
    plt.figure(figsize=(ncols*4, nrows*4))
    gs = mpl.gridspec.GridSpec(nrows=nrows, ncols=ncols)

    for i, plot_dict in enumerate(plot_dicts):
        irow = int(i/ncols)
        icol = i % ncols
        ax = plt.subplot(gs[irow, icol])
        ax.set_title(plot_dict['hue'])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)        
        g = sns.scatterplot(data=df, x=x, y=y, s=s, lw=lw, **plot_dict)
        # g.legend(loc='center left', bbox_to_anchor=(0, 1.5), ncol=2)
        ax.set_aspect('equal')
    return g

# Correlation matrix
def plot_clustermap(adata, col_color='celltype'):
    if col_color == 'celltype':
        lut = {'Negative': 'grey', 'B+': 'blue', 'R+': 'red', 'B+R+': 'purple'}
        col_colors = adata.var.celltype.map(lut)
    elif col_color in adata.var.columns:
        sr = adata.var[col_color]
        sr = (sr - sr.min()) / (sr.max() - sr.min())
        unique_vals = sr.unique()
        palette = mpl.colormaps['Blues']
        lut = dict(zip(unique_vals, palette(unique_vals)))
        col_colors = sr.map(lut)
    else:
        col_colors = None
    X = pd.DataFrame(data=adata.X, columns=adata.var_names)
    clustergrid = sns.clustermap(X.corr(), center=0, square=True, 
                                 xticklabels=False, col_colors=col_colors)
    clustergrid.fig.suptitle(session_title(adata), va='bottom')

# Quality control
def plot_trigger_check(sync, tlim):
    if type(tlim) is str:
        if tlim == 'start':
            t0 = sync.scan.t.iloc[0]
        elif tlim == 'end':
            t0 = sync.scan.t.iloc[-1]
        tlim = (t0 - 0.25, t0 + 0.25)
        
    idt = (sync.abf.t >= tlim[0]) & (sync.abf.t < tlim[1])

    ax1 = plt.subplot(211)
    plt.plot(sync.abf.t[idt], sync.abf.Pockels[idt])
    plt.ylabel('Pockels')

    plt.subplot(212, sharex=ax1)
    plt.plot(sync.abf.t[idt], sync.abf.ScanImage[idt])
    plt.scatter(sync.scan.t, np.ones(len(sync.scan.t)))
    plt.ylabel('ScanImage Triggers')
    plt.xlabel('Time (s)')
    plt.xlim(*tlim)

def plot_top_slice(adata):
    img = imread(adata.uns['path']['meanRef']['filter1'][0])
    img = img / img.max() * 2
    plt.imshow(img)

def plot_top_slices(adatas, region='PPC'):
    if region is not None:
        adatas = [adata for adata in adatas if adata.uns['metadata']['region']==region]
    nplots = len(adatas)
    ncols = 4
    nrows = int(np.ceil(nplots/ncols))
    plt.figure(figsize=(ncols*3, nrows*3))
    gs = mpl.gridspec.GridSpec(nrows=nrows, ncols=ncols)
    for i, adata in enumerate(adatas):
        icol = i % ncols
        irow = int(i/ncols)
        ax = plt.subplot(gs[irow, icol])
        plot_top_slice(adata)

def plot_top_slice_corrmat(adatas, region='PPC'):
    if region is not None:
        adatas = [adata for adata in adatas if adata.uns['metadata']['region']==region]
    images_flat = [imread(adata.uns['path']['meanRef']['filter1'][0])[:, :, 1].flatten() for adata in adatas]
    columns = [adata.uns['metadata']['date'] for adata in adatas]
    df = pd.DataFrame(np.stack(images_flat, axis=1), columns=columns)
    sns.heatmap(df.corr(), vmin=0, vmax=1)

def plot_F_raw(adata, var_key=None, plane=1, max_ncells=10):
    plt.figure()
    F = np.load(adata.uns['path']['F_npy'].format(plane=1), allow_pickle=True)
    if var_key is not None:
        var_idx = an.fetch_index(adata.var, var_key)
    var_names = adata.var_names[var_idx]
    cells = [unpack_var_name(var_name) for var_name in var_names]
    isources = [cell[1] for cell in cells if cell[0]==plane]
    isources = isources[:max_ncells]
    gs = mpl.gridspec.GridSpec(nrows=len(isources), ncols=1)
    for i, isource in enumerate(isources):
        ax = plt.subplot(gs[i, 0])
        plt.plot(F[isource], lw=0.5)

# Specificity  
def plot_channels(sessions, x, y, hue=None, xlim=None, ylim=None, vline=None, hline=None, log=True, palette=None):
    if type(sessions) is not list: sessions = [sessions]
    var = an.aggr_var(sessions)
    if log:
        num = var._get_numeric_data()
        num[num < 0] = 0
        
    sns.set_theme(style='ticks')
    # Load the planets dataset and initialize the figure
    g = sns.JointGrid(data=var, x=x, y=y, hue=hue, palette=palette)
    g.plot_joint(sns.scatterplot)
    g.plot_marginals(sns.histplot, stat='probability', element='step')
    if log:
        g.ax_joint.set(yscale='log', xscale='log')

    if xlim: g.ax_joint.set_xlim(xlim)
    if ylim: g.ax_joint.set_ylim(ylim)
    if vline: g.ax_joint.axvline(vline)        
    if hline: g.ax_joint.axhline(hline)

def specificity_vs_cutoff(sessions, maxval=None):
    if type(sessions) is not list: sessions = [sessions]
    var = an.aggr_var(sessions)
    var = var[var['B_spatial_corr'] > 0.5]
    cutoff = np.arange(0, 1, 0.05).astype(np.float64)
    
    maxval_computed = var['B'].dropna().sort_values()[-3:].mean()
    if maxval is None: 
        maxval = maxval_computed
    print('max value: %.0f' %maxval_computed)
    df = pd.DataFrame(index=cutoff, columns=['specificity', 'count'])
    df.index.name = 'cutoff'
    for cutoffi in cutoff:
        blue_cells = var[var['B'] >= cutoffi * maxval]
        df.loc[cutoffi, 'specificity'] = blue_cells['R+'].mean()
        df.loc[cutoffi, 'count'] = len(blue_cells)
    return df.reset_index()

def plot_specificity_vs_cutoff(sessions, red_threshold=300, maxval=None):
    df = specificity_vs_cutoff(sessions, maxval=maxval)
    f = plt.figure(figsize=(6, 6))
    gs = f.add_gridspec(2, 1, height_ratios=[1, 2])
    
    axbar = f.add_subplot(gs[0, 0])
    axbar.bar(df['cutoff'], df['count'], width=0.05, color='none', edgecolor='C0')
    axbar.set_yscale('log')
    
    axline = f.add_subplot(gs[1, 0], sharex=axbar)
    axline.plot(df['cutoff'], df['specificity'])

# Photostimulation
def imshow_raw_alignment(adata, start_ind=10, end_ind=-10, vmax=0.02, figsize=(4, 4)):
    raw_tifs = glob.glob(os.path.join(adata.uns['path']['raw2P_dir'], 'session_*.tif'))
    img = np.zeros((512, 512, 3))
    img[:, :, 0] = imread(raw_tifs[9]).mean(axis=0) / (2**15)
    img[:, :, 1] = imread(raw_tifs[-9]).mean(axis=0) / (2**15)

    plt.figure(figsize=figsize)
    plt.imshow(img/vmax)
    show_targets(adata, group=1, color='white')
    save_file = os.path.join(adata.uns['path']['preprocessed_dir'], '2P', 'raw_alignment.tif')
    imwrite(save_file, img)

def show_targets(adata, group=1, color='white', arrow=(10, 10), offset=(-5, -5), width=0.01, head_width=4, annotate_target='nearest_source'):
    # Show targets
    targets = adata.uns['photostim_targets']
    targets['isource'] = targets['nearest_source'].apply(lambda source: int(source.split('source')[-1]))
    if group is not None:
        targets = targets[targets['group']==group].sort_values('isource')
    dx, dy = arrow
    ax = plt.gca()
    for i, (_, target) in enumerate(targets.iterrows()):
        plt.arrow(target['x']-dx+offset[0], target['y']-dy+offset[0], dx, dy, width=width, length_includes_head=True, head_starts_at_zero=True, head_width=head_width, facecolor=color, edgecolor=color)
        if annotate_target:
            annotation = target[annotate_target]
            ax.annotate(annotation, (targets['x'].iloc[i], targets['y'].iloc[i]-10), c=color, size=5)
        
def plot_target_locations(adata, filter_id, channel=None, vmax=None, plane=1, **kwargs):
    # Read image
    img_file = adata.uns['path']['meanRef'][filter_id][plane-1]
    img = imread(img_file)
    img = img / (2**15)
    img[img<0] = 0
    
    # Clear channels other than the one to be displayed
    if channel is not None:
        for channeli in [1, 2, 3]:
            if channeli != channel:
                img[:, :, channeli-1] = 0
    
    # Show image
    img2 = img.copy()
    img2[:, :, 0] = img[:, :, 2]
    img2[:, :, 2] = img[:, :, 0]
    plt.imshow(img2/vmax)
    ax = plt.gca()
    ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
    # Show targets
    if 'group' in adata.uns['photostim_targets'].columns:
        show_targets(adata, group=1, color='hotpink', **kwargs)
        show_targets(adata, group=2, color='grey', **kwargs)
    else:
        show_targets(adata, group=None, color='white', **kwargs)
        
def plot_target_locations_allchannels(adata, vmax=[1, 1, 1], plane=1, **kwargs):
    plt.figure(figsize=(15, 5))
    plt.suptitle(session_title(adata))
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=3)
    
    keys = [
        dict(filter_id='filter1', channel=2),
        dict(filter_id='filter2', channel=3),
        dict(filter_id='filter3', channel=1)]
    for i, key in enumerate(keys):
        ax = plt.subplot(gs[0, i])
        if key['filter_id'] in adata.uns['path']['meanRef'].keys():
            plot_target_locations(adata, key['filter_id'], channel=key['channel'], vmax=vmax[i], plane=plane, **kwargs)

def plot_trace_with_stim_marker(adata, tlim=None, marker_delay_s=1, stim_on_s=1, signal='dF', ylim=None, yticks=None, figsize=None, scatter_kwargs={}, plot_kwargs={}):
    if signal == 'dF':
        Xstim = adata.layers['dF']
    elif signal == 'X':
        Xstim = adata.X
    print(Xstim.shape)
    photostim_sequence = adata.uns['photostim_sequence']
    if figsize is None:
        figsize = (14, Xstim.shape[1])
    plt.figure(figsize=figsize)
    plt.suptitle(session_title(adata), y=1.2)
    gs = mpl.gridspec.GridSpec(nrows=Xstim.shape[1], ncols=1)

    t = adata.obs['t'].to_numpy()
    marker_delay_inds = int(marker_delay_s / adata.obs['dt'].mean())
    stim_t = photostim_sequence.loc[photostim_sequence['group']==1, 't']
    stim_inds = np.array([np.where(t>=ti)[0][0] for ti in stim_t])
    ctl_t = photostim_sequence.loc[photostim_sequence['group']==2, 't']
    ctl_inds = np.array([np.where(t>=ti)[0][0] for ti in ctl_t])

    for i, cell in enumerate(Xstim.T):
        if i == 0:
            ax0 = ax = plt.subplot(gs[i, 0])
        else:
            ax = plt.subplot(gs[i, 0], sharex=ax0, zorder=1)
        if i < len(Xstim.T)-1:
            ax.xaxis.set_visible(False)
        ax.set_ylabel(f'Target {i+1}', ha='right', rotation=0)

        plt.plot(t, cell, **plot_kwargs, zorder=2)
        dt = adata.obs['dt'].mean()
        idt = stim_inds+marker_delay_inds
        idt_idx = idt < len(cell)
        idt = idt[idt_idx]
        ymin = ylim[0] if ylim is not None else 0
        for ti in stim_t[idt_idx]:
            plt.fill_betweenx(y=[ymin, cell.max()], x1=ti, x2=ti+stim_on_s, color='pink', lw=0, alpha=0.7, zorder=1)
        # plt.scatter(stim_t[idt_idx]+marker_delay_inds*dt, cell[idt], c='hotpink', zorder=3, label='photostimulation', **scatter_kwargs)
        
        idt = ctl_inds+marker_delay_inds
        idt_idx = idt < len(cell)
        idt = idt[idt_idx]
        for ti in ctl_t[idt_idx]:
            plt.fill_betweenx(y=[ymin, cell.max()], x1=ti, x2=ti+stim_on_s, color='grey', lw=0, alpha=0.3, zorder=1)
        # plt.scatter(ctl_t[idt_idx]+marker_delay_inds*dt, cell[idt], c='grey', zorder=3, label='control', **scatter_kwargs)
        # if i == 0:
        #     plt.legend(bbox_to_anchor=(0.9, 0.5, 0.3, 0.3))
        if not ylim is None:
            plt.ylim(ylim)

        if not yticks is None:
            plt.yticks(yticks)
        
    if tlim is not None:
        plt.xlim(tlim)

def plot_photostim_triggered_traces(adata, mean=True, ylim=(0, 2.), photostim_range=None):
    photostim_sequence = adata.uns['photostim_sequence'].copy()
    if photostim_range is not None:
        photostim_sequence = photostim_sequence.iloc[photostim_range[0]:photostim_range[1]]
    if len(photostim_sequence) == 0:
        return
    trig = an.trigger_inds(adata.obs, trigger_t=photostim_sequence['t'], t_range=(-2, 5), safe=False)
    var_key={'stim_group1': True}
    var_idx = an.fetch_index(adata.var, var_key)
    var_names = adata[:, var_idx].var_names
    Xtrig = an.fetch_X(adata[:, var_idx].layers['dF'], trig['idyx'], norm=False)
    
    plt.figure(figsize=(16, 2))
    plt.suptitle(session_title(adata), y=1.6)
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=len(Xtrig))
    groups = [1, 2]
    color = {1: 'red', 2: 'darkgrey'}
    for icell in range(len(Xtrig)):
        ax = plt.subplot(gs[0, icell])
        if icell == 0:
            plt.ylabel('Deconvolved Activity')
            plt.xlabel('Time (s)')
        else:
            ax.yaxis.set_visible(False)
        for igroup, group in enumerate(groups):            
            Xi = Xtrig[icell, photostim_sequence['group']==group]
            if mean:
                lineplot(Xi, xticks=trig['t'], color=color[group])
            else:
                plt.plot(trig['t'], Xi.T, color=color[group], lw=0.25)
        ax.set_ylim(ylim)
        ax.set_title(var_names[icell])

def plot_photostim_targets_Fraw(adata):
    F_npy = adata.uns['path']['F_npy'].format(plane=1)
    F = np.load(F_npy, allow_pickle=True)
    icells = [int(var_name.split('_source')[-1]) for var_name in adata[:, adata.var['stim_group1']].var_names]
    
    plt.figure(figsize=(14, len(icells)))
    plt.suptitle(session_title(adata))
    gs = mpl.gridspec.GridSpec(nrows=len(icells), ncols=1)
    for i, cell in enumerate(F[icells]):
        if i == 0:
            ax0 = ax = plt.subplot(gs[i, 0])
        else:
            ax = plt.subplot(gs[i, 0], sharex=ax0, zorder=1)
        ax.set_ylabel(f'Target {i+1}', ha='right', rotation=0)
        
        plt.plot(cell)

def plot_influence_hist(adata, xlim=(-2, 2)):
    influence = adata.var['influence_value']

    bins = np.arange(np.floor(influence.min()), np.ceil(influence.max()), 0.05)
    
    ax = plt.gca()
    ax.hist(influence[~adata.var['stim_group1']], bins=bins, label='non-targets', color='grey')
    # ax.hist(influence[adata.var['stim_group1']], bins=bins, label='stim group', color='red')
    ax.hist(influence[~adata.var['stim_group1'] & (adata.var['influence_pvalue'] < 0.01)], bins=bins, label='p<0.01', color='tab:orange')
    # ax.hist(influence[~adata.var['stim_group1'] & adata.var['B+']], bins=bins, label='BFP+', color='tab:blue')

    ax.set_yscale('log')
    plt.xlim(xlim)
    ax.axvline(0, c='black', lw=0.5, ls='--')
    plt.legend()
    plt.ylabel('Count')
    plt.xlabel('Influence (std)')
    plt.suptitle(session_title(adata))

def plot_var_errorbar1(var_cat, x, y, var_key=None, color='black', bins=4, s=5, lw=1):
    var_idx = an.fetch_index(var_cat, var_key)
    var = var_cat[var_idx]
    cut = pd.cut(var[x], bins=bins)
    grouped = var[y].groupby(cut)
    yi = grouped.mean()
    bootstrap = grouped.apply(bootstrap_ci)
    stat_test = grouped.apply(scipy.stats.wilcoxon)
    print(var_key, stat_test)
    yerr = np.abs(np.array([[r.low, r.high] for r in bootstrap]).T - np.array(yi))
    xi = yi.index.dtype.categories.mid
    plt.plot(xi, yi, color=color, label=str(var_key), lw=lw)
    plt.scatter(xi, yi, color=color, s=s)
    plt.errorbar(xi, yi, yerr=yerr, color=color, lw=lw, capsize=2, capthick=0.25)
    plt.xlabel(x)
    plt.ylabel(y)
    return grouped

def plot_influence_v_distance_deprecated(adatas, x='dist_to_closest_target_after_50um', y='influence', var_keys=None, legend=True, s=5, lw=1, bins=[0, 100, 200, 300], min_trials=0):
    var_keys = copy.deepcopy(var_keys)
    var_cat = pd.concat([adata.var for adata in adatas], axis=0, ignore_index=True)
    var_cat = var_cat[(~var_cat[y].isna()) & (var_cat['ntrials_stim'] > min_trials) & (var_cat['ntrials_ctl'] > min_trials)]
    kwargs = dict(
        var_cat = var_cat,
        x = x,
        y = y,
        bins = bins)

    for var_key in var_keys:
        color = var_key.pop('color', 'black')
        plot_var(**kwargs, var_key=var_key, color=color, s=s, lw=lw)
    if legend:
        plt.legend()

    ax = plt.gca()
    ax.axhline(0, c='black', lw=0.25, ls='--')

def plot_var_errorbar(var_cat, x, y, var_keys=None, legend=True, s=5, lw=1, bins=[0, 100, 200, 300]):
    var_keys = copy.deepcopy(var_keys)
    kwargs = dict(
        var_cat = var_cat,
        x = x,
        y = y,
        bins = bins)

    for var_key in var_keys:
        color = var_key.pop('color', 'black')
        plot_var_errorbar1(**kwargs, var_key=var_key, color=color, s=s, lw=lw)
    if legend:
        plt.legend()

    ax = plt.gca()
    ax.axhline(0, c='black', lw=0.25, ls='--')

# Photostimulation triggered images
def plot_influence_image(adata, frames, t, tlim=(0.5, 1.5), tlim0=(-1, 0), figsize=(6, 6), vrange=0.5, **kwargs):
    # Compute dF/F
    frames = np.moveaxis(frames, 2, 0)
    frames -= frames.min()
    F0 = frames[(t>=tlim0[0]) & (t<tlim0[1])].mean(axis=0)
    F = frames[(t>=tlim[0]) & (t<tlim[1])].mean(axis=0)
    dF = (F-F0) / F0
    
    # Plot dF/F map
    plt.figure(figsize=figsize)
    plt.suptitle(session_title(adata))
    plt.imshow(dF, cmap='seismic', vmin=-vrange, vmax=vrange)
    
    # Show targets
    show_targets(adata, group=1, color='hotpink', **kwargs)
    show_targets(adata, group=2, color='grey', **kwargs)

    plt.colorbar(shrink=0.7)

def plot_photostim_frame_difference(adata, conditions, key='avgMov', **kwargs):
    if 'photostim_frames' not in adata.uns['path'].keys():
        adata.uns['path']['photostim_frames'] = os.path.join(adata.uns['path']['preprocessed_dir'], 'photostim', 'photostim_frames_{condition}.mat')
        
    mats = {condition: loadmat(adata.uns['path']['photostim_frames'].format(condition=condition)) for condition in conditions}
    frames = mats[conditions[0]][key] - mats[conditions[1]][key]
    assert (mats[conditions[0]]['t'] == mats[conditions[1]]['t']).all()
    t = mats[conditions[0]]['t'][0]
    plot_influence_image(adata, frames, t, **kwargs)

# Generate correlation maps
def generate_correlation_image_plane(adata, var_key=None, show=False, save=True, plane=1, meanref_ichan=1, corr_ichan=0):
    # Initialize 3 channel image
    md = adata.uns['metadata']
    img = np.zeros((md['Ly'], md['Lx'], 3))

    # Add meanref to green channel
    meanref_file = adata.uns['path']['meanRef']['filter1'][plane-1]
    meanref = imread(meanref_file)[:, :, meanref_ichan]
    img[:, :, 1] = meanref / meanref.max()

    # Color cells by correlation to mean of cells identified by var_key
    var_idx = an.fetch_index(adata.var, var_key)
    var_names = adata[:, var_idx].var_names
    planes = np.array([int(var_name.split('plane')[-1].split('_source')[0]) for var_name in var_names])
    var_names = var_names[planes==plane]
    var_mean = adata[:, var_names].X.mean(axis=1)
    
    stat = np.load(adata.uns['path']['stat_npy'].format(plane=plane), allow_pickle=True)
    for var_name in var_names:
        icell = int(var_name.split('_source')[-1])
        stati = stat[icell]
        corr = np.corrcoef(adata[:, var_name].X[:, 0], var_mean)[1, 0]
        img[stati['ypix'], stati['xpix'], corr_ichan] = (stati['lam'] / stati['lam'].max()) * corr
    
    if show:
        plt.figure(figsize=(6, 6))
        plt.suptitle(session_title(adata))
        plt.imshow(img)
    
    if save:
        index = meanref_file.find('.tif')
        save_file = meanref_file[:index] + '_varcorr.tif'
        imwrite(save_file, img)
    return img

def generate_correlation_images(adata, var_key=None, show=False, save=True, meanref_ichan=1, corr_ichan=0):
    nslices = adata.uns['metadata']['nslices']
    for plane in range(1, nslices+1):
        generate_correlation_image_plane(adata, plane=plane, var_key=var_key, show=show, save=save, meanref_ichan=meanref_ichan, corr_ichan=corr_ichan)

# Plot images
def tile_images(images, ncols=5, vmin=0, vmax=300, cmap=None, title=None):
    nplots = len(images)
    nrows = int(np.ceil(nplots/ncols))
    plt.figure(figsize=(ncols*4, nrows*4))
    gs = mpl.gridspec.GridSpec(nrows=nrows, ncols=ncols)
    if title is not None:
        assert len(title) == len(images)
    for i, img in enumerate(images):
        irow = int(i/ncols)
        icol = i%ncols
        ax = plt.subplot(gs[irow, icol])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        plt.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
        if title is not None:
            ax.set_title(title[i])
