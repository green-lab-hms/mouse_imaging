from mouse_imaging import config

def default_ops(imaging=True, env=None, maze=None): # env is the mouse_imaging module computing maze variables, e.g. 'tmaze'. None skips this step
    ops = {}
    
    # directories, set per system in the mouse_imaging config file (see config.py)
    ops['raw_root'] = config.get_path('raw_root')
    ops['preprocessed_root'] = config.get_path('derived_root')

    
    ops['ball_diam_cm'] = 20.32
    ops['ball_circumference_cm'] = ops['ball_diam_cm'] * np.pi
    ops['cm_per_virmen_unit'] = ops['ball_circumference_cm'] / 86
    ops['imaging'] = imaging
    ops['env'] = env
    ops['maze'] = maze
    
    ops['triggers'] = []
    ops['is_photostim'] = False
    
        
    if imaging:
        # ops['convnet_mat'] = f"/home/{user}/code/source_classifier_syt/convNet_l23_171216.mat"

        ops['filter1'] = 'filter1'
        ops['filter2']= 'filter2'

        ops['functional_chan'] = 'G'

        # var
        ops['dilation'] = {'cell_dilation': 0, 'background_dilation': [2, 0]}
        ops['demix'] = None
        ops['min_dist_to_edge'] = 10
        ops['max_val'] = 0.95
        ops['triggers'].extend(['ScanImage'])

        # suite2p: overrides to suite2p.default_settings(), merged in preprocess.py so suite2p and torch are only imported there
        ops['suite2p_settings'] = {
            'tau': 1.25, # timescale of gcamp to use for deconvolution
            'diameter': [6, 6], # check
            'run': {'multiplane_parallel': False}, # True sends planes to a remote server over ssh, not local parallelism
            'io': {'delete_bin': True},
            'registration': {'align_by_chan2': False,
                             'do_bidiphase': True},
            'detection': {'threshold_scaling': 2.0, # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
                          'chan2_threshold': 0.25,
                          'cellpose_chan2': True},
            'extraction': {'neuropil_extract': True,
                           'neuropil_coefficient': 0.7},
            'dcnv_preprocess': {'baseline': 'maximin'},
        }

        ops['do_corrmat'] = True
        ops['do_nneighbor_graph'] = True
        ops['do_umap'] = True
        ops['do_leiden'] = True
        
    # vr
    ops['is_vr_playback'] = False
    ops['do_median_trajectory'] = False
    ops['virmen_mat_columns'] = ['world_id', 'dx', 'dy', 'dh', 'x', 'y', 'h_int', 'inITI', 'reward', 'dt', 'lick', 'trial']
    ops['sync_gains'] = {'Ball_pitc': -121 * ops['cm_per_virmen_unit'],
                'Ball_roll': -1,
                'Ball_yaw': -1,
                'Reward': 1,}

    ops['sync_offsets'] = {'Reward': 0}

    ops['sync_labels'] = {'Reward': 'sync_reward'}
    return ops

import numpy as np
import itertools
import pandas as pd


# Call cell types
def label_cell(cell):
    pos = sorted(cell.index[cell.index.str.endswith('+')])
    available_labels = [''.join(combi) for combi in itertools.combinations(pos + [''], r=2)] + ['Negative']
    label = ''.join([posi for posi in pos if cell[posi]])
    if not label: label = 'Negative'
    return label

def call_celltypes_BR(var, ops):
    var = var.copy()
    B_thresh = var['B'].sort_values().dropna().tail(ops['top_n_B']).mean() * ops['min_B_rel']
    var['B+'] = (var['B'] > B_thresh)

    R_thresh = var['R'].sort_values().dropna().tail(ops['top_n_R']).mean() * ops['min_R_rel']
    var['R+'] = (var['R'] > R_thresh) | (var['R_spatial_corr'] > ops['min_R_spatial_corr'])

    var['celltype'] = var.apply(label_cell, axis=1).astype('category')
    return var

def call_celltypes_BR_ChRmine(var, ops):
    var = var.copy()
    B_thresh = var['B'].sort_values().dropna().tail(ops['n_top_B']).mean() * ops['min_B_rel']
    var['B+'] = (var['B'] > B_thresh)

    R_thresh = var['R'].sort_values().dropna().tail(ops['n_top_R']).mean() * ops['min_R_rel']
    var['R+'] = (var['R'] > R_thresh) | (var['R_spatial_corr'] > ops['min_R_spatial_corr'])

    # var.drop(['B+Sst44+', 'Sst44+'], axis=1, errors='ignore', inplace=True)
    var['celltype'] = var.apply(label_cell, axis=1).astype('category')
    var['Sst44+'] = var['B+']
    var['R+Sst44-'] = var['R+'] & (~var['B+'])
    var['Nonlabeled'] = (~var['R+']) & (~var['B+'])
    return var

def call_celltypes_B(adatas, thresh=0.15):
    for adata in adatas:
        md = adata.uns['metadata']
        topBmean = adata.var.B.sort_values().dropna()[-3:].mean()
        adata.var['B+'] = (adata.var['B'] > topBmean * thresh)
        adata.var.drop(['B+Sst44+', 'Sst44+'], axis=1, errors='ignore', inplace=True)
        adata.var['celltype'] = adata.var.apply(label_cell, axis=1).astype('category')
        adata.var['Sst44+'] = adata.var['B+']

def celltype_stats_BR(adatas):
    df = pd.DataFrame(columns=['R+', 'B+', 'B+R+', 'Negative', 'B+R+/B+', 'mouse', 'region'])
    rows = []
    for adata in adatas:
        row = pd.Series({'R+': (adata.var.celltype=='R+').sum(),
              'B+': (adata.var.celltype == 'B+').sum(),
              'B+R+': (adata.var.celltype == 'B+R+').sum(),
              'Negative': (adata.var.celltype == 'Negative').sum(),
              'B+R+/B+': (adata.var.celltype == 'B+R+').sum() / adata.var['B+'].sum(),
              'B+R+/R+': (adata.var.celltype == 'B+R+').sum() / adata.var['R+'].sum(),
              'mouse': adata.uns['metadata']['mouse'],
              'region': adata.uns['metadata']['region'],
              'date': pd.to_datetime(adata.uns['metadata']['date'], yearfirst=True)})
        rows.append(row)
    df = pd.DataFrame.from_records(rows)        
    return df

def celltype_stats_BR_ChRmine(adatas):
    df = pd.DataFrame(columns=['B-R+', 'B+R+', 'R-B+','B+', 'Negative', 'B+R+/B+', 'mouse', 'region'])
    rows = []
    for adata in adatas:
        row = pd.Series({
              'B-R+': (adata.var.celltype=='R+').sum(),
              'B+R+': (adata.var.celltype=='B+R+').sum(),
              'B+': (adata.var['B+']).sum(),
              'Negative': (adata.var.celltype == 'Negative').sum(),
              'B+R+/B+': (adata.var.celltype == 'B+R+').sum() / adata.var['B+'].sum(),
              'mouse': adata.uns['metadata']['mouse'],
              'region': adata.uns['metadata']['region']})
        rows.append(row)
    df = pd.DataFrame.from_records(rows) 
    return df      

# Processing
def preprocess_activity(adata, sigma_s=0.25):
    print('Preprocessing activity.')
    import scipy.ndimage

    if 'dcnv' not in adata.layers.keys():
        adata.layers['dcnv'] = adata.X

    sigma = sigma_s / adata.obs['dt'].mean()
    adata.layers['dcnv_0.25sigma'] = scipy.ndimage.gaussian_filter1d(adata.layers['dcnv'], sigma=sigma, axis=0)
    cell_max = np.percentile(adata.layers['dcnv_0.25sigma'], 99, axis=0)
    
    # Remove noise
    adata = adata[:, cell_max>0].copy()
    cell_max = cell_max[cell_max>0]

    adata.layers['dcnv_0.25sigma_norm'] = adata.layers['dcnv_0.25sigma'] / cell_max
    adata.layers['dcnv_norm'] = adata.layers['dcnv'] / cell_max

    return adata
    
def add_corrmat(adata, layers):
    # Compute correlation matrix
    for layer in layers:
        print(f'Computing correlation matrix on {layer}.')
        adata.varp[f'corr_{layer}'] = pd.DataFrame(adata.layers[layer], columns=adata.var_names).corr()

def add_nneighbor_graph(adata, layers, n_neighbors=10):
    from sklearn.neighbors import kneighbors_graph
    # Compute nearest neighbor graph matrix
    for layer in layers:
        print(f'Computing nearest neighbor graph on {layer}.')
        adata.varp[f'nearest_neighbor_{n_neighbors}_{layer}'] = kneighbors_graph(adata.layers[layer].T, n_neighbors=n_neighbors)

def add_umap(adata, layers):
    import mouse_imaging.analysis as an
    # Compute UMAP
    for layer in layers:
        print(f'Computing umaps on {layer}.')
        adata.obs[[f'umap_x_{layer}', f'umap_y_{layer}']] = an.umap_Xtime(adata.layers[layer])
        adata.var[[f'umap_x_{layer}', f'umap_y_{layer}']] = an.umap_Xcell(adata.layers[layer])

def add_leiden_clustering(adata, layers):
    import mouse_imaging.analysis as an
    # Compute leiden clustering
    for layer in layers:
        print(f'Computing leiden clustering on {layer}.')
        adata.var[f'leiden_{layer}'] = an.leiden_Xcell(adata.layers[layer]).membership

def process_SstCreRFP_Sst44nlsBFP(adata, do_corrmat=True, do_nneighbor_graph=True, do_umap=True, do_leiden=True):
    adata.var = call_celltypes_BR(adata.var, adata.uns['ops'])
    adata = preprocess_activity(adata, sigma_s=0.25)
    layers = ['dcnv_norm', 'dcnv_0.25sigma_norm']
    if do_corrmat:
        add_corrmat(adata, layers)
    if do_nneighbor_graph:
        add_nneighbor_graph(adata, layers, n_neighbors=10)
    if do_umap:
        add_umap(adata, layers)
    if do_leiden:
        add_leiden_clustering(adata, layers)

    print('Computing celltype mean activity.')
    celltype_idx = {'Sst44': adata.var['celltype']=='B+R+',
                'Sst': adata.var['celltype']=='R+',
                'Nonlabeled': adata.var['celltype']=='Negative'}
    layers_norm = ['dcnv_norm', 'dcnv_0.25sigma_norm']
    for label, var_idx in celltype_idx.items():
        for layer in layers_norm:
            adata.obs[f'{label}_{layer}'] = adata[:, var_idx].layers[layer].mean(axis=1)
    return adata

def process_ChRmine_Sst44nlsBFP(adata, do_corrmat=False, do_nneighbor_graph=False, do_umap=False, do_leiden=False):
    adata.var = call_celltypes_BR_ChRmine(adata.var, adata.uns['ops'])
    adata = preprocess_activity(adata, sigma_s=0.25)
    layers = ['dcnv', 'dcnv_0.25sigma']
    if do_corrmat:
        add_corrmat(adata, layers)
    if do_nneighbor_graph:
        add_nneighbor_graph(adata, layers, n_neighbors=10)
    if do_umap:
        add_umap(adata, layers)
    if do_leiden:
        add_leiden_clustering(adata, layers)

    print('Computing celltype mean activity.')
    celltype_idx = {'Sst44': adata.var['Sst44+'],
                'R+Sst44-': adata.var['R+Sst44-'],
                'Nonlabeled': adata.var['Nonlabeled']}
    layers_norm = ['dcnv_norm', 'dcnv_0.25sigma_norm']
    for label, var_idx in celltype_idx.items():
        for layer in layers_norm:
            adata.obs[f'{label}_{layer}'] = adata[:, var_idx].layers[layer].mean(axis=1)
    return adata

def save_adatas(adatas):
    for adata in adatas:
        adata.write(adata.uns['path']['adata_h5ad'])
        
def process_adatas(adatas, call_celltypes=None, save=True):
    for i, adata in enumerate(adatas):
        print('Processing {mouse} {date} {session}.'.format(**adata.uns['metadata']))
        if call_celltypes is not None:
            call_celltypes(adata)
        adatas[i] = adata = add_variables(adata)
        if save:
            adata.write(adata.uns['path']['adata_h5ad'])
