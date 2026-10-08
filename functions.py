import numpy as np
import scipy.ndimage.morphology
import pyabf
import h5py
import scipy.io
import pandas as pd
import pickle
import scipy.stats

# Trigger index operations
def get_inds_start_stop(inds, min_len=0, keep_ends=False):
    diff = np.diff(inds.astype(np.int8))
    start_inds = np.where(diff>0)[0] + 1
    stop_inds = np.where(diff<0)[0] + 1
    if len(start_inds) and len(stop_inds):
        if len(stop_inds) == 0:
            stop_inds = np.array([len(inds)-1])
        elif len(start_inds) == 0:
            start_inds = np.array([0])

        if stop_inds[0] <= start_inds[0]:
            if keep_ends:
                start_inds = np.insert(start_inds, 0, 0)
            else:
                stop_inds = stop_inds[1:]

        if not len(stop_inds):
            return np.array([]), np.array([])

        if start_inds[-1] >= stop_inds[-1]:
            if keep_ends:
                stop_inds = np.append(stop_inds, len(inds)-1)
            else:
                start_inds = start_inds[:-1]
        if min_len:
            pulse_len = stop_inds - start_inds
            start_inds = start_inds[pulse_len >= min_len]
            stop_inds = stop_inds[pulse_len >= min_len]
        
        return start_inds, stop_inds
    else:
        return np.array([]), np.array([])

def rising_idx(arr, min_dist=None, direction='rising', return_bool=False):
    """
    Get indices triggered on changes in arr.
    """
    assert direction in ['rising', 'falling', 'both']
    if direction == 'rising':
        inds = np.where(np.diff(arr.astype(int)) > 0)[0] + 1
    elif direction == 'falling':
        inds = np.where(np.diff(arr.astype(int)) < 0)[0] + 1
    elif direction == 'both':
        inds = np.where(np.abs(np.diff(arr.astype(int))) > 0)[0] + 1
    if min_dist and len(inds):
        inds = inds[np.insert(np.where(np.diff(inds) > min_dist)[0] + 1, 0, 0)]

    if return_bool:
        idx = np.zeros(len(arr)).astype(bool)
        idx[inds] = True
        return idx
    else:
        return inds

def first_index(inds, since):
    first_inds = []
    for i, since_ind in enumerate(since):
        ind_diff = inds - since_ind
        first_ind_diff = ind_diff[ind_diff>0].min()
        first_inds.append(since_ind + first_ind_diff)
    first_inds = np.array(first_inds)
    return first_inds

def idyx(idy, window, dt=None):
    """
    Returns 2D array of indices for each epoch.
    """
    window = np.array(window)
    if dt:
        window = window / dt
    window = window.astype(int)
    idx = np.arange(*window)
    _idyx = np.tile(idx, (len(idy), 1)) + idy[:, None]
    if dt:
        ttrig = idx * dt
        return _idyx, ttrig
    else:
        return _idyx

def trigger_idyx(sr, window=(-5, 10), min_dist_s=0.5):
    t = sr.index # assumes index is time
    dt = np.diff(t[:100]).mean()
    idy = rising_idx(sr, min_dist=min_dist_s/dt)
    idx = np.arange(window[0]/dt, window[1]/dt).astype(int)
    idyx = np.tile(idx, (len(idy), 1)) + idy[:, None]
    t = idx * dt
    return idyx, t

def index_safe(arr, idx, mode='omit'):
    arr = arr.astype(float)
    out_of_bounds = np.zeros_like(idx).astype(bool)
    out_of_bounds[idx >= len(arr)] = True
    out_of_bounds[idx < 0] = True

    # if mode == 'omit':
    #     idx = idx[out_of_bounds.any(axis=tuple(range(1, arr.ndim)))]
    # else:

    idx = idx.copy()
    idx[idx < 0] = 0
    idx[idx >= len(arr)] = 0
    result = arr[idx]
    result[out_of_bounds] = np.nan
    return result

def peak_onset(arr, peak_thresh, onset_thresh):
    peaks = rising_idx(arr > peak_thresh)
    if len(peaks) == 0:
        return np.array([])
    onsets = rising_idx(arr > onset_thresh)
    diff = np.tile(onsets, (len(peaks), 1)) - peaks[:, None]
    def closest_onset(row):
        neg = row[row<0]
        if not len(neg):
            return 0
        else:
            return np.argmax(neg)
    onsets_idx = np.apply_along_axis(closest_onset, axis=1, arr=diff)
    return onsets[onsets_idx]

def binary_extend_forward(idx, extension):
    idx_extended = np.zeros(len(idx)).astype(bool)
    inds = rising_idx(idx)
    for ind0 in inds:
        idx_extended[ind0:ind0+extension] = True
    return idx_extended

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
    
# Circular operations functions
def wrap_h(h):
    h = ( (h + np.pi) % (2 * np.pi) ) - np.pi
    # h[h>np.pi] -= 2*np.pi
    # h[h<-np.pi] += 2*np.pi
    return h

def circmean(x, **kwargs):
    return wrap_h(scipy.stats.circmean(x, **kwargs))

def nancircmean(x, **kwargs):
    return circmean(x, nan_policy='omit', **kwargs)

def nancircstd(x, **kwargs):
    return scipy.stats.circstd(x, nan_policy='omit', **kwargs)

# Interpolation
def interp_sr(sr, index_new, kind='linear'):
    if sr.name == 'h':
        f = scipy.interpolate.interp1d(sr.index.mid, np.unwrap(sr.values), kind=kind, bounds_error=False, fill_value=np.nan)
        result = wrap_h( f(index_new) )
    else:
        f = scipy.interpolate.interp1d(sr.index.mid, sr.values, kind=kind, bounds_error=False, fill_value=np.nan)
        result = f(index_new)
    return result

def interp_df(df, index_new):
    df = df.select_dtypes(include='float64')
    df_interp = pd.DataFrame(index=index_new)
    for col in df.columns:
        df_interp[col] = interp_sr(df[col], index_new)
    return df_interp

# Statistics
def bootstrap_ci(x, confidence_level=0.95, n_resamples=20):
    result = scipy.stats.bootstrap((x,), statistic=np.nanmean, confidence_level=confidence_level, n_resamples=n_resamples)
    return result.confidence_interval

# Other functions
def dilate(mask, dilation):
    if dilation > 0:
        mask = scipy.ndimage.morphology.binary_dilation(mask, iterations=dilation)
    elif dilation < 0:
        mask = scipy.ndimage.morphology.binary_erosion(mask, iterations=-dilation)
    return mask

def delay_signal(signal, delay):
    delay = delay * -1
    sr = pd.Series(0, index=signal.index)
    delay_inds = np.abs(delay)
    if delay > 0:
        sr[delay_inds:] = signal[:-delay_inds]
    else:
        sr[:-delay_inds] = signal[delay_inds:]
    return sr

def less_than_window(x, max_val, window_s, dt):
    kernel = np.ones(int(window_s / dt))
    x_conv = np.convolve(x > max_val, kernel, mode='same') == 0
    return x_conv
    
def crosscorr(x, y, dt, tlim=(-5, 5)):
    
    x /= x.std()
    y /= y.std()
    
    # Compute crosscorr
    xlen = x.shape[0]
    xcorr = scipy.signal.correlate(x, y) / xlen
    
    # Compute time base
    lags = np.arange(-xlen + 1, xlen) * dt
    assert len(lags) == len(xcorr)

    idx = (lags >= tlim[0]) & (lags < tlim[1])
    return xcorr[idx], lags[idx]

# Images
def crop_img(img, center, crop=(60, 60)):
    Ly, Lx = img.shape[:2]
    y, x = map(int, center)
    dx, dy = map(lambda x: int(x/2), crop)
    top = max(0, y-dy)
    bottom = min(Ly, y+dy)
    left = max(0, x-dx)
    right = min(Lx, x+dx)
    cropped = img[top:bottom, left:right]
    return cropped

# Vector operations
def vector(start, end):
    result = np.diff(np.stack([start, end]), axis=0)[0]
    return result

def point2line(point, start, end):
    line_vec = vector(start, end)
    point_vec = vector(start, point)
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)    
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = line_vec * t
    nearest_vec = vector(nearest, point_vec)
    dist = np.linalg.norm(nearest_vec)
    nearest = nearest + start
    return (nearest, dist)

# Linear regression on lower part of data
def select_pixels_to_regress(x, y, xbins=20, x_left=0.02, x_right=0.5, y_left=0.2, y_right=0.5, add_origin=False, xmin=-0.2):
    # Remove negative outliers
    idx = x > xmin
    x = x[idx]
    y = y[idx]

    x_sel = np.array([])
    y_sel = np.array([])

    # Left side
    idx = x < np.percentile(x, x_left*100)
    x2 = x[idx]
    y2 = y[idx]
    idx2 = y2 < np.percentile(y2, y_left*100)
    x_sel = np.concatenate([x_sel, x2[idx2]])
    y_sel = np.concatenate([y_sel, y2[idx2]])

    # Right side
    idx = x > np.percentile(x, x_right*100)
    x2 = x[idx]
    y2 = y[idx]
    idx2 = y2 < np.percentile(y2, y_right*100)
    x_sel = np.concatenate([x_sel, x2[idx2]])
    y_sel = np.concatenate([y_sel, y2[idx2]])
        
    return x_sel, y_sel

# Stats
def bootstrap(data, statistic=np.nanmean, axis=0, ci=0.95, n_resamples=10):
    assert len(data.shape) == 2
    result = scipy.stats.bootstrap(data, statistic, axis=axis, confidence_level=ci, n_resamples=n_resamples,)
    return np.array(result.confidence_interval)

# Importing /saving data
def load_pickle(filename):
    with open(filename, 'rb') as fh:
        obj = pickle.load(fh)
    return obj

def save_pickle(obj, filename):
    with open(filename, 'wb') as fh:
        pickle.dump(obj, fh)

def import_abf(abf_filename):
    abf = pyabf.ABF(abf_filename)
    df = pd.DataFrame(index=abf.sweepX, data=abf.data.T, columns=abf.adcNames)
    df['t'] = abf.sweepX
    return df

def import_mat(mat_filename):
    mat = scipy.io.loadmat(mat_filename)
    return mat

def import_h5(h5_filename, rename_columns={}):
    f = h5py.File(h5_filename, 'r')
    data_key = list(f.keys())[1]
    arr = np.array(f[data_key]['analogScans']) / 1000 # units are in mV
    columns = [col.decode() for col in list(f['header']['AIChannelNames'])]
    df = pd.DataFrame(arr.T, columns=columns)
    sampling_rate = list(f['header']['AcquisitionSampleRate'])[0][0]
    df['t'] = np.arange(len(df)) / sampling_rate
    df['dt'] = 1 / sampling_rate
    df = df.rename(rename_columns)
    return df
