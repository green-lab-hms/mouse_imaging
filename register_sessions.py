import numpy as np
import pandas as pd
import scipy.stats
import tempfile
import os
import json

import caiman as cm
from caiman.motion_correction import MotionCorrect, sliding_window
from normcorre import compute_pwrigid_transform, apply_pwrigid_transform
import cv2
from tifffile import imread

import anndata
from mouse_imaging.session import define_path, get_metadata, maze_id, adata_to_key
from functions import save_pickle, load_pickle

from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import matplotlib as mpl

# Compile data
def load_meanRefs(keys, iplane=0, filter_id='filter1', ichannel=1):
    arr2ds = []
    for key in keys:
        path = define_path(**key)
        img = imread(path['meanRef'][filter_id][iplane])[:, :, ichannel]
        arr2ds.append(img)
    arr3d = np.stack(arr2ds)
    return arr3d

def source_images(key, iplane, filter_id='filter1', ichannel=1, smooth_pix=0):
    path = define_path(**key)
    metadata = get_metadata(path)
    img = imread(path['meanRef'][filter_id][iplane])[:, :, ichannel]
    stat = np.load(path['stat_npy'].format(plane=iplane+1), allow_pickle=True)
    source_imgs = np.zeros((len(stat), metadata['Ly'], metadata['Lx']))
    if smooth_pix: mask0 = np.zeros((metadata['Ly'], metadata['Lx']))
    for i, stati in enumerate(stat):
        if smooth_pix:
            mask = mask0.copy()
            mask[stati['ypix'], stati['xpix']] = 1
            mask = binary_dilation(mask, iterations=smooth_pix).astype(float)
            mask = gaussian_filter(mask, smooth_pix)
            source_imgs[i] = img * mask
        else:
            source_imgs[i, stati['ypix'], stati['xpix']] = img[stati['ypix'], stati['xpix']]
    return source_imgs


# Match sources
def match_source_to_ref(source, source_b, ref, ref_b, min_corr=0.7, ret_corr=False):
    overlap = source_b & ref_b
    if overlap.sum() == 0 and not ret_corr:
        return np.nan
    iref = np.argmax(overlap.sum(axis=-1).sum(axis=-1))    
    corr = scipy.stats.pearsonr(source.flatten(), ref[iref].flatten())[0]
    if ret_corr:
        return corr
    else:
        if corr > min_corr:
            return iref
        else:
            return np.nan

def invert_mapping(map_, n):
    map_inv = [np.nan] * n
    for key, val in enumerate(map_):
        if not np.isnan(val):
            map_inv[int(val)] = key
    return map_inv

def match_sources_to_ref(sources, ref, min_corr=0.7, ret_corr=False, keep_source='mean'):
    # Precompute binary sources
    ref_b = ref > 0
    sources_b = sources > 0

    # Compute source-to-ref mapping
    src_to_ref = [match_source_to_ref(source, source_b, ref, ref_b, min_corr=min_corr, ret_corr=ret_corr)
                 for source, source_b in zip(sources, sources_b)]

    # Invert source-to-ref mapping
    ref_to_src = invert_mapping(src_to_ref, len(ref))
    
    # Add unmapped sources
    src_ids = range(len(sources))
    src_ids_unmapped = list(set(src_ids) - set(ref_to_src))
    mapping = pd.Series(ref_to_src+src_ids_unmapped)

    # Replace source images
    ref_new = ref.copy()
    if keep_source == 'first':
        # Do not replace
        pass
    elif keep_source == 'last':
        mapped_inds = np.where(np.isnan(ref_to_src) == False)[0]
        ref_new[mapped_inds] = sources[ref_to_src[mapped_inds].astype(int)]
    elif keep_source == 'mean':
        # Weighs older sources exponentially lower
        mapped_inds = np.where(np.isnan(ref_to_src) == False)[0]
        sources_mapped_new = sources[ref_to_src[mapped_inds].astype(int)]
        sources_mapped_old = ref[mapped_inds]
        ref_new[mapped_inds] = np.stack([sources_mapped_old, sources_mapped_new]).mean(axis=0)
    else:
        raise ValueError('Argument keep_source must equal "first" , "last" or "mean".')
    
    # Append unmatched sources
    ref_new = np.concatenate([ref_new, sources[src_ids_unmapped]], axis=0)
    
    assert mapping.max() == len(sources)-1
    assert len(mapping) == len(ref_new)

    return mapping, ref_new

def match_sources(sources, min_corr=0.5, keep_source='mean'):
    keys = list(sources.keys())
    df = pd.DataFrame()

    # Initialize reference with first session
    ref = sources[keys[0]]
    df[keys[0]] = range(len(ref))

    # Recursively build reference by matching to each session
    for key in keys[1:]:
        print(f'Matching sources from {key}.')
        sr, ref = match_sources_to_ref(sources[key], ref, min_corr=min_corr, keep_source=keep_source)
        df = pd.concat([df, sr.rename(key)], axis=1)

    return df

def label_sources(df, plane):
    prefix = f'plane{plane}_source'
    def to_cell_index(sr):
        return sr.apply(lambda x: np.nan if np.isnan(x) else f'{prefix}{int(x)}')
    df = df.apply(to_cell_index)
    df.index = f'plane{plane}_usource' + df.index.astype(str)
    return df

# Merge sessions
def match_var_names(adata, sourcemap):
    key = adata_to_key(adata)
    date = adata.uns['metadata']['date']
    # ref_to_adata = sourcemap[str(key)].dropna()
    ref_to_adata = sourcemap[str(key)].dropna()
    adata_to_ref = pd.Series(ref_to_adata.index.values, index=ref_to_adata)
    var_names = adata.var_names.map(adata_to_ref)
    return var_names

def mean_vars_col(adatas, col):
    df = pd.DataFrame()
    for i, adata in enumerate(adatas):
        df = pd.concat([df, adata.var[col]], axis=1)
    
    sr = df.astype(float).mean(axis=1, skipna=True)
    return sr

def mean_vars(adatas, var_names, columns):
    var = pd.DataFrame(index=var_names)
    for col in columns:
        var[col] = mean_vars_col(adatas, col)
    return var

def concat_sessions(adatas, sourcemap):
    # Assert all from same region
    regions = [adata.uns['metadata']['region'] for adata in adatas]
    assert len(set(regions)) == 1

    # Match var_names
    adatas = adatas.copy()
    for adata in adatas:
        adata.var_names = match_var_names(adata, sourcemap)
    
    # Concatenate along obs axis, append session id to obs
    sessions = [f"{adata.uns['metadata']['date']}_{adata.uns['metadata']['session']}" for adata in adatas]
    adata_m = anndata.concat(adatas, join='outer', axis=0, keys=sessions, label='session', fill_value=np.nan, merge=None)
    adata_m.obs_names_make_unique()  
    
    # Append maze to obs
    mazes = [maze_id(adata.uns['path']) for adata in adatas]    
    for session, maze in zip(sessions, mazes):
        adata_m.obs.loc[adata_m.obs.session==session, 'maze'] = maze
    
    # Take mean of vars, excluding UMAP projection
    columns = [col for col in adatas[0].var.columns if not col.startswith('Xumap')]
    adata_m.var = mean_vars(adatas, adata_m.var_names, columns)
    
    return adata_m

# Plot source images for quality control
def load_stat(mouse, date, plane=1):
    path = define_path(mouse=mouse, date=date)
    stat = np.load(path['stat_npy'].format(plane=plane), allow_pickle=True)
    return stat

def load_stats(mouse, dates, plane=1):
    stats = [load_stat(mouse, date, plane=plane) for date in dates]
    return stats

def stat2img(stati, Ly=512, Lx=512):
    img = np.zeros((Ly, Lx))
    img[stati['ypix'], stati['xpix']] = stati['lam']
    return img
    
def stat2border(stati, **kwargs):
    img = stat2img(stati, **kwargs) > 0
    border = scipy.ndimage.morphology.binary_dilation(img, iterations=1)
    border[img] = False
    return border

def crop_slice(stati, crop=(100, 100)):
    y_margin, x_margin = map(int, np.array(crop)/2)
    y_center, x_center = map(int, stati['med'])
    y_slice = slice(y_center-y_margin, y_center+y_margin)
    x_slice = slice(x_center-x_margin, x_center+x_margin)
    return y_slice, x_slice

def crop(img, stati, crop=(100, 100)):
    y_slice, x_slice = crop_slice(stati, crop=crop)
    return img[y_slice, x_slice]

def add_border(img, border):
    img = img.copy()
    img[border] = img.max()
    return img

def plot_registered_sources(keys, sourcemap_i, filter_id='filter1', ichannel=1, vmax=99.5):
    path = define_path(mouse=mouse, date=dates[0])
    md = get_metadata(path)
    planes = range(1, md['nslices']+1)
    stats = {plane: {date: stat for date, stat in zip(dates, load_stats(mouse, dates, plane=plane))} for plane in planes}
    imgs = {plane: {date: img for date, img in zip(dates, load_meanRefs(mouse, dates, iplane=plane-1, filter_id=filter_id, ichannel=ichannel))} for plane in planes}

    plt.figure(figsize=(3*len(dates), 3))
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=len(dates))
    for idate, (date, source_id) in enumerate(zip(sourcemap_i.index, sourcemap_i)):
        ax = plt.subplot(gs[0, idate]) 
        if type(source_id) is str:
            plane = int(source_id.split('plane')[-1].split('_')[0])
            isource = int(source_id.split('source')[-1])
            img = imgs[plane][date].copy()
            stati = stats[plane][date][isource]
            img = add_border(img, stat2border(stati))
            img_cropped = crop(img, stati)
            
            if len(img_cropped.flatten()):
                vmax_abs = np.percentile(img_cropped.flatten(), vmax)
                ax.imshow(img_cropped, vmax=vmax_abs)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.set_title(f'{date}\n{source_id}')
        else:
            ax.axis('off')
            ax.set_title(f'{date}\nno match')


# Scripts
def filepath(keys, plane=None, min_corr=None, makedir=False):
    dates = [key['date'] for key in keys]
    path = define_path(**keys[0])
    metadata = get_metadata(path)

    mouse = keys[0]['mouse']
    filedir = os.path.join(path['preprocessed_root'], mouse, 'multisession_registration')
    if makedir:
        os.makedirs(filedir, exist_ok=True)

    if min_corr is None:
        mincorr_txt = ''
    else:
        mincorr_txt = f'_mincorr{min_corr}'
    filename = f'source_mapping_{mouse}_{metadata["region"]}_{"-".join(dates)}_mincorr{min_corr}' + '_plane{plane}.pickle'
    if plane is None:
        filename = filename.format(plane='').replace('_plane', '') 
    elif type(plane) is int:
        filename = filename.format(plane=plane)
        
    filepath = os.path.join(filedir, filename)
    return filepath

def dates_to_keys(mouse, dates):
    keys = []
    for date in dates:
        if date[0] == '{':
            key = dict(mouse=mouse, **json.loads(date))
        else:
            key = dict(mouse=mouse, date=date, session='session_1')
        keys.append(key)
    return keys

def register_plane(keys, plane, filter_id_source='filter1', ichannel_source=1, filter_id_ref='filter1', ichannel_ref=1, min_corr=0.7, keep_source='mean'):
    path = define_path(**keys[0])
    metadata = get_metadata(path)
    iplane = plane - 1

    mouse = keys[0]['mouse']
    print(f'Processing plane {mouse} plane {plane}.')
    # Load meanRefs
    print('Loading mean references...')
    ref_images = load_meanRefs(keys, iplane=iplane, filter_id=filter_id_ref, ichannel=ichannel_ref)

    # Compute transform for each day
    print('Computing piecewise rigid transform...')
    mc = compute_pwrigid_transform(ref_images, iref=int(len(keys)/2))

    # Compile source images
    print('Compiling source images...')
    sources_orig = [source_images(key, iplane, filter_id=filter_id_source, ichannel=ichannel_source) for key in keys]

    # Apply transform to source images
    print('Applying transform to source images...')
    sources_reg = {str(key): apply_pwrigid_transform(orig, mc.x_shifts_els[i], mc.y_shifts_els[i], mc.overlaps, mc.strides)
                  for i, (key, orig) in enumerate(zip(keys, sources_orig))}

    # Match sources across days
    print('Matching sources across days...')
    sourcemap = match_sources(sources_reg, min_corr=min_corr, keep_source=keep_source)
    sourcemap = label_sources(sourcemap, plane)

    # Write mapping
    print('Saving source mapping...')
    sourcemap_file = filepath(keys, plane, min_corr=min_corr, makedir=True)
    save_pickle(sourcemap, sourcemap_file)

    return sourcemap

def concatenate_planes(keys, min_corr=0.7):
    path = define_path(**keys[0])
    metadata = get_metadata(path)

    sourcemap_concat = pd.concat([
        load_pickle(filepath(keys, plane=plane, min_corr=min_corr)) 
        for plane in range(1, metadata['nslices']+1)], axis=0)
    
    # Write
    filepath_concat = filepath(keys, plane=None, min_corr=min_corr)
    save_pickle(sourcemap_concat, filepath_concat)

    return sourcemap_concat

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Map sources across sessions.')
    parser.add_argument('--mouse', required=True, type=str)
    parser.add_argument('--dates', required=True, nargs='+')
    parser.add_argument('--plane', required=True, type=str)
    parser.add_argument('--filter_id_source', required=False, type=str, default='filter1')
    parser.add_argument('--filter_id_ref', required=False, type=str, default='filter1')
    parser.add_argument('--ichannel_source', required=False, type=int, default=1)
    parser.add_argument('--ichannel_ref', required=False, type=int, default=1)
    parser.add_argument('--min_corr', required=False, type=float, default=0.6)
    parser.add_argument('--keep_source', required=False, type=str, default='mean')
    args = parser.parse_args()
    arg_dict = vars(args)

    keys = [{'mouse': rg_dict.pop('mouse'), 'date': date.split('_')[0], 'session': 'session_' + date.split('_')[1]} for date in arg_dict.pop('dates')]
    # keys = dates_to_keys(arg_dict.pop('mouse'), arg_dict.pop('dates'))

    if args.plane == 'concatenate':
        concatenate_planes(keys, min_corr=args.min_corr)
    elif args.plane.isnumeric():
        args.plane = int(args.plane)
        register_plane(keys, **arg_dict)