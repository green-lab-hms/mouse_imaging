"""
Quality-control report for a preprocessed session, saved as qc_report.pdf next to adata.h5ad.

Summarizes the session (mouse, date, session, recording time, cell and R+ cell counts) and lists warnings
for registration problems and drift, using what suite2p saves per plane in ops.npy:
    xoff/yoff     rigid shift of each frame (lateral motion)
    corrXY        registration correlation of each frame with the reference image (drops with z drift or brightness changes)
    badframes     frames suite2p excluded because of outlier shifts
    regDX         residual motion after registration, from principal components of the registered movie
plus each plane's fluorescence (F.npy), the suite2p log (run.log) and the sync check from the anndata step.
"""
import re, json, time, textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage
import matplotlib
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from mouse_imaging import session as sess
from mouse_imaging import options

def _smooth(x, n):
    """Running median over n samples."""
    return scipy.ndimage.median_filter(np.asarray(x, dtype=float), size=max(int(n), 1), mode='nearest')

def _start_end_change(x, frac=0.1):
    """Fractional change from the first to the last `frac` of a trace, using medians."""
    n = max(int(len(x) * frac), 1)
    start, end = np.median(x[:n]), np.median(x[-n:])
    return (end - start) / abs(start) if start != 0 else np.nan

def suite2p_log_messages(run_log):
    """Unique ERROR/WARNING lines from suite2p's run.log, without timestamps."""
    messages = []
    try:
        with open(run_log) as fh:
            for line in fh:
                if re.search(r'ERROR|WARN', line):
                    message = re.sub(r'^[\d\-: ,]+\[\w+\]\s*', '', line).strip()
                    if message not in messages:
                        messages.append(message)
    except FileNotFoundError:
        messages.append(f'No suite2p log found at {run_log}')
    return messages

# Report styling (dataviz reference palette, light mode for print)
STYLE = {
    'surface': '#ffffff',
    'panel': '#f4f3f0',        # stat tiles, table header
    'grid': '#e6e5e1',         # hairline gridlines and rules
    'text': '#0b0b0b',
    'text_secondary': '#52514e',
    'text_muted': '#8a8985',
    'series_1': '#2a78d6',     # blue: x shift, single series
    'series_1_light': '#d6e6fa',
    'series_2': '#eb6834',     # orange: y shift
    'series_2_light': '#f9d9cb',
    'good': '#0ca30c',
    'warning': '#fab219',
    'critical': '#d03b3b',
}
RC = {
    'font.family': 'DejaVu Sans', 'font.size': 8, 'text.color': STYLE['text'],
    'axes.edgecolor': STYLE['grid'], 'axes.linewidth': 0.8, 'axes.labelcolor': STYLE['text_secondary'],
    'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlecolor': STYLE['text'],
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'axes.grid.axis': 'y', 'axes.axisbelow': True,
    'grid.color': STYLE['grid'], 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'xtick.color': STYLE['text_muted'], 'ytick.color': STYLE['text_muted'], 'xtick.labelcolor': STYLE['text_secondary'],
    'ytick.labelcolor': STYLE['text_secondary'], 'xtick.major.size': 0, 'ytick.major.size': 0,
    'figure.facecolor': STYLE['surface'], 'axes.facecolor': STYLE['surface'], 'legend.frameon': False,
}
WARN_ICON = '⚠'   # ⚠
OK_ICON = '✓'     # ✓
PAGE = (8.5, 11)

# Per-plane metrics: label, format, and the test that flags a value
METRICS = [
    ('frames', 'Volumes', '{:,.0f}', None),
    ('rois_detected', 'ROIs detected', '{:,.0f}', None),
    ('suite2p_cells', 'Cells (suite2p)', '{:,.0f}', None),
    ('selected_cells', 'Cells (selected)', '{:,.0f}', None),
    ('R+_cells', 'R+ cells', '{:,.0f}', None),
    ('max_shift_px', 'Max rigid shift', '{:.1f} px', None),
    ('lateral_drift_px', 'Slow lateral drift', '{:.1f} px', lambda v, t: v > t['max_lateral_drift_px']),
    ('badframe_frac', 'Bad frames', '{:.1%}', lambda v, t: v > t['max_badframe_frac']),
    ('corr_change', 'Registration quality, start → end', '{:+.0%}', lambda v, t: v < -t['max_corr_drop']),
    ('brightness_change', 'Cell brightness, start → end', '{:+.0%}', lambda v, t: abs(v) > t['max_brightness_change']),
    ('regDX_px', 'Residual motion (regDX)', '{:.2f} px', lambda v, t: v > t['max_regdx_px']),
]
LIMITS = {
    'lateral_drift_px': lambda t: f"≤ {t['max_lateral_drift_px']} px",
    'badframe_frac': lambda t: f"≤ {t['max_badframe_frac']:.0%}",
    'corr_change': lambda t: f"≥ −{t['max_corr_drop']:.0%}",
    'brightness_change': lambda t: f"within ±{t['max_brightness_change']:.0%}",
    'regDX_px': lambda t: f"≤ {t['max_regdx_px']} px",
}

def compute_qc(adata, ops=None):
    """
    QC metrics and warnings for one session.
    Returns a dict with 'info', 'stats', 'planes' (DataFrame), 'flags' (DataFrame of bool), 'thresholds', 'warnings' and 'traces'.
    """
    if ops is None:
        ops = options.default_ops()
    thresh = ops['qc']
    md, path = adata.uns['metadata'], adata.uns['path']
    fs = md['volume_rate']
    window = thresh['drift_window_s'] * fs

    planes, traces = [], {}
    for plane in range(md['nslices']):
        ops_plane = np.load(path['ops_npy'].format(plane=plane), allow_pickle=True).item()
        stat = np.load(path['stat_npy'].format(plane=plane), allow_pickle=True)
        iscell = np.load(path['iscell_npy'].format(plane=plane))[:, 0].astype(bool)
        F = np.load(path['F_npy'].format(plane=plane))
        var_plane = adata.var[adata.var['plane'] == plane]
        sources = [sess.unpack_var_name(name)[1] for name in var_plane.index]

        xoff, yoff, corr = ops_plane['xoff'], ops_plane['yoff'], ops_plane['corrXY']
        xs, ys, cs = _smooth(xoff, window), _smooth(yoff, window), _smooth(corr, window)
        # Brightness: each selected cell's F relative to its median, averaged over cells
        F_sel = F[sources] if len(sources) else F[iscell]
        brightness_raw = np.nanmean(F_sel / np.median(F_sel, axis=1, keepdims=True), axis=0)
        brightness = _smooth(brightness_raw, window)
        regDX = ops_plane.get('regDX')

        planes.append({
            'plane': plane,
            'frames': len(xoff),
            'rois_detected': len(stat),
            'suite2p_cells': int(iscell.sum()),
            'selected_cells': len(var_plane),
            'R+_cells': int(var_plane['redcell'].sum()) if 'redcell' in var_plane else np.nan,
            'max_shift_px': float(max(np.abs(xoff).max(), np.abs(yoff).max())),
            'lateral_drift_px': float(max(np.ptp(xs), np.ptp(ys))),
            'badframe_frac': float(np.mean(ops_plane.get('badframes', np.zeros(len(xoff), bool)))),
            'corr_change': float(_start_end_change(cs)),
            'brightness_change': float(_start_end_change(brightness)),
            'regDX_px': float(np.max(regDX[:, 3])) if regDX is not None and len(regDX) else np.nan,
        })
        traces[plane] = dict(t=np.arange(len(xoff)) / fs, xoff=xoff, yoff=yoff, xs=xs, ys=ys, corr=corr, cs=cs,
                             brightness_raw=brightness_raw, brightness=brightness,
                             meanImg=ops_plane['meanImg'], meanImg_chan2=ops_plane.get('meanImg_chan2'))
    planes = pd.DataFrame(planes).set_index('plane')

    # Flags and warnings
    flags = pd.DataFrame(False, index=planes.index, columns=planes.columns)
    for key, _, _, test in METRICS:
        if test is not None:
            flags[key] = planes[key].apply(lambda v: bool(np.isfinite(v) and test(v, thresh)))
    # One warning per check, listing every plane that crossed the limit with its value
    plane_checks = [  # metric, value format, message, short label for the dashboard
        ('lateral_drift_px', '{:.1f} px', f"Slow lateral drift above {thresh['max_lateral_drift_px']} px (smoothed rigid shift) in {{planes}}.", 'lateral drift'),
        ('badframe_frac', '{:.1%}', f"Frames flagged as bad by suite2p registration above {thresh['max_badframe_frac']:.0%} in {{planes}}.", 'bad frames'),
        ('corr_change', '{:+.0%}', 'Registration quality dropped from start to end in {planes}. Possible z drift or loss of image quality.', 'z drift'),
        ('brightness_change', '{:+.0%}', "Cells' mean fluorescence changed from start to end in {planes}. Possible z drift, bleaching or laser power change.", 'brightness'),
        ('regDX_px', '{:.2f} px', f"Residual motion after registration (suite2p regDX) above {thresh['max_regdx_px']} px in {{planes}}.", 'residual motion'),
    ]
    warnings, warning_labels = [], []
    def warn(label, message):
        warning_labels.append(label)
        warnings.append(message)
    for key, fmt, message, label in plane_checks:
        flagged = planes.index[flags[key]]
        if len(flagged):
            # non-breaking spaces keep each 'plane N (value)' on one line when the report wraps text
            values = [f'plane\u00a0{plane}\u00a0({fmt.format(planes.loc[plane, key]).replace(" ", chr(0xa0))})' for plane in flagged]
            listed = values[0] if len(values) == 1 else ', '.join(values[:-1]) + ' and ' + values[-1]
            warn(label, message.format(planes=listed))
    if planes['frames'].max() - planes['frames'].min() > 1:
        warn('frame counts', f"Planes have different frame counts: {planes['frames'].to_dict()}.")
    qc_uns = adata.uns.get('qc', {})
    if 'sync_minus_suite2p_volumes' in qc_uns and abs(qc_uns['sync_minus_suite2p_volumes']) > thresh['max_volume_mismatch']:
        warn('sync mismatch', f"Sync file has {qc_uns['sync_volumes']} imaging volumes but suite2p has {qc_uns['suite2p_volumes']} "
                        f"({qc_uns['sync_minus_suite2p_volumes']:+d}). Imaging and behavior alignment may be off.")
    n_red = int(adata.var['redcell'].sum()) if 'redcell' in adata.var else 0
    red_frac = n_red / adata.n_vars if adata.n_vars else np.nan
    red_flag = bool(red_frac > thresh['max_red_frac'])
    if red_flag:
        warn('R+ cells', f"{red_frac:.0%} of cells are R+ (suite2p redcell). The red cell threshold or method is probably not separating cells.")
    for message in suite2p_log_messages(str(path['suite2p_dir']) + '/run.log'):
        warn('suite2p log', f'suite2p log: {message}')

    # filter* stacks (filters step): alignment to the session and z offset
    filters, notes = _filter_qc(adata), []
    # filter* stacks that were uploaded but are missing from the AnnData (filters step not run, failed, or skipped)
    try:
        raw_filters = list(sess.define_path(md['mouse'], md['date'], md['session'], ops=ops).get('filter_raw_dirs', {}))
    except Exception:
        raw_filters = []
    missing = [name for name in raw_filters if name not in filters]
    if missing:
        warn('filter stacks', f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} in the raw data but not in the AnnData. "
                              "Run the filters and anndata steps; if it was already run, check the job log for that stack's error.")
    for name, f in filters.items():
        corr = np.array(f['corr_after'], dtype=float)
        low = [p for p, c in enumerate(corr) if not np.isfinite(c) or c < thresh['min_filter_corr']]
        if low:
            listed = ', '.join(f'plane\u00a0{p}\u00a0({corr[p]:.2f})' for p in low)
            warn('filter alignment', f"{name} matches the session image poorly after alignment in {listed} (correlation of "
                                     f"high-passed green images, limit {thresh['min_filter_corr']}). Wrong field of view or z plane?")
    # Recordings whose last TIFF was cut off (MATLAB crash): suite2p read a repaired copy (tifrepair.py)
    repair_json = Path(str(path['preprocessed_dir'])) / 'repaired_tifs' / 'repair.json'
    if repair_json.exists():
        r = json.loads(repair_json.read_text())
        notes.append(f"The last TIFF ({Path(r['file']).name}) was cut off after {r['pages_complete']} of {r['pages_listed']} frames, "
                     f"probably because MATLAB crashed. suite2p used {r['volumes_kept']:,} whole volumes and dropped the last "
                     f"{r['frames_dropped']} frames; a sync mismatch at the end of the recording is expected.")
    # One note per z offset, naming every stack taken at it
    by_offset = {}
    for name, f in filters.items():
        offsets = tuple(sorted(set(float(z) for z in f['z_offset_um'])))
        if any(abs(z) > 0 for z in offsets):
            by_offset.setdefault(offsets, []).append(name)
    for offsets, names in by_offset.items():
        notes.append(f"{' and '.join(names)} {'was' if len(names) == 1 else 'were'} taken {' / '.join(f'{z:+g}' for z in offsets)} µm "
                     "in z from the session planes (ScanImage z positions); planes are matched one to one.")

    t = adata.obs['t'].to_numpy()
    imaging_min = planes['frames'].max() / fs / 60
    aligned_min = (t[-1] - t[0] + np.median(np.diff(t))) / 60
    stats = [  # headline numbers: (value, label, detail, flagged)
        (f'{aligned_min:.1f} min', 'Behavior time', f'of {imaging_min:.1f} min imaging', False),
        (f'{adata.n_vars:,}', 'Cells', f"of {int(planes['rois_detected'].sum()):,} ROIs detected", False),
        (f'{n_red:,}', 'R+ cells', f'{red_frac:.0%} of cells', red_flag),
        (f"{md['nslices']}", 'Planes', f"+{md['nflyback']} flyback, {fs:.2f} volumes/s", False),
    ]
    info = {
        'Mouse': md['mouse'],
        'Date': md['date'],
        'Session': md['session'],
        'Region': md.get('region') or '—',
        'Maze': md.get('maze') or '—',
        'Channels': ', '.join(md['filter1']['channels']),
        'Frame size': f"{md['Ly']} × {md['Lx']} px",
        'Volume rate': f'{fs:.2f} Hz',
        'Imaging time': f"{imaging_min:.1f} min ({planes['frames'].max():,} volumes)",
        'Aligned to behavior': f"{aligned_min:.1f} min ({adata.n_obs:,} volumes)",
        'Cells': f"{adata.n_vars:,} selected, {int(planes['suite2p_cells'].sum()):,} by suite2p",
        'ROIs detected': f"{int(planes['rois_detected'].sum()):,}",
        'R+ cells': f'{n_red:,} ({red_frac:.0%} of selected cells)',
    }
    if filters:
        info['Filter stacks'] = ', '.join(f"{name} ({', '.join(f['channels'])}{', %d px' % f['Ly'] if f['Ly'] != md['Ly'] else ''})"
                                          for name, f in filters.items())
    return dict(info=info, stats=stats, planes=planes, flags=flags, thresholds=thresh, warnings=warnings, warning_labels=warning_labels,
                filters=filters, notes=notes, session_g=[tr['meanImg'] for tr in traces.values()],
                traces=traces, n_cells=int(adata.n_vars), n_red=n_red, behavior_min=float(aligned_min), imaging_min=float(imaging_min))

def _filter_qc(adata):
    """Alignment info of each filter* stack from adata.uns['filters'], plus its mean images for the report."""
    import tifffile
    from mouse_imaging import functions as fc
    out = {}
    path = adata.uns['path']
    for name, f in (adata.uns.get('filters') or {}).items():
        f = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in f.items()}
        f['shift_yx'] = [list(map(int, s)) for s in np.reshape(f.get('shift_yx', []), (-1, 2))]
        for key in ['corr_before', 'corr_after', 'z_offset_um', 'channels']:
            f[key] = list(np.atleast_1d(f.get(key, [])))
        out_dir = Path(str((path.get('filter_dirs') or {}).get(name, Path(str(path['preprocessed_dir'])) / name)))
        try:
            fmd = json.loads((out_dir / 'metadata.json').read_text())
            stack = tifffile.imread(out_dir / 'mean.tif').astype(float).reshape(fmd['nslices'], len(fmd['channels']), fmd['Ly'], fmd['Lx'])
        except Exception as e:
            print(f'QC: could not read the {name} mean images ({type(e).__name__}); showing its alignment only.')
            f['images'] = {}
            out[name] = f
            continue
        scale = int(f.get('scale', 1))
        # Same-size stacks are shown shifted into session coordinates; larger ones at full resolution
        f['images'] = {(plane, ch): (fc.shift_img(stack[plane, c], f['shift_yx'][plane]) if scale == 1 else stack[plane, c])
                       for plane in range(len(f['shift_yx'])) for c, ch in enumerate(fmd['channels'])}
        out[name] = f
    return out

# ---- Rendering ----

def _header(fig, qc, title, page, npages):
    info = qc['info']
    fig.text(0.06, 0.965, f"{info['Mouse']}  ·  {info['Date']}  ·  {info['Session']}", fontsize=8, color=STYLE['text_secondary'])
    fig.text(0.06, 0.94, title, fontsize=15, weight='bold')
    fig.add_artist(matplotlib.lines.Line2D([0.06, 0.94], [0.928, 0.928], color=STYLE['grid'], lw=0.8))
    fig.text(0.06, 0.02, 'mouse_imaging QC report', fontsize=7, color=STYLE['text_muted'])
    fig.text(0.94, 0.02, f'{page} / {npages}', fontsize=7, color=STYLE['text_muted'], ha='right')

def _status_badge(fig, x, y, n_warnings):
    color = STYLE['warning'] if n_warnings else STYLE['good']
    icon = WARN_ICON if n_warnings else OK_ICON
    label = f'{n_warnings} warning' + ('s' if n_warnings != 1 else '') if n_warnings else 'No warnings'
    fig.text(x, y, icon, fontsize=12, color=color, ha='right', va='center', weight='bold')
    fig.text(x + 0.008, y, label, fontsize=10, weight='bold', va='center', ha='left')

def _summary_page(pdf, qc, npages):
    fig = plt.figure(figsize=PAGE)
    _header(fig, qc, 'Preprocessing QC', 1, npages)
    _status_badge(fig, 0.80, 0.946, len(qc['warnings']))

    # Stat tiles
    x0, w, gap, top, h = 0.06, 0.205, 0.02, 0.905, 0.085
    for i, (value, label, detail, flagged) in enumerate(qc['stats']):
        x = x0 + i * (w + gap)
        fig.add_artist(matplotlib.patches.FancyBboxPatch((x, top - h), w, h, boxstyle='round,pad=0,rounding_size=0.008',
                                                         transform=fig.transFigure, facecolor=STYLE['panel'], edgecolor='none'))
        fig.text(x + 0.012, top - 0.02, label, fontsize=8, color=STYLE['text_secondary'])
        fig.text(x + 0.012, top - 0.052, value, fontsize=17, weight='bold')
        fig.text(x + 0.012, top - 0.074, detail, fontsize=7, color=STYLE['text_secondary'])
        if flagged:
            fig.text(x + w - 0.012, top - 0.02, WARN_ICON, fontsize=10, color=STYLE['warning'], ha='right', weight='bold')

    # Session details, two columns
    y = 0.78
    fig.text(0.06, y, 'Session', fontsize=10, weight='bold')
    items = list(qc['info'].items())
    half = (len(items) + 1) // 2
    for col, chunk in enumerate([items[:half], items[half:]]):
        yy = y - 0.026
        for key, val in chunk:
            fig.text(0.06 + col * 0.44, yy, key, fontsize=8, color=STYLE['text_secondary'])
            fig.text(0.20 + col * 0.46, yy, str(val), fontsize=8)
            yy -= 0.019
    y = y - 0.026 - half * 0.019 - 0.02

    # Per-plane table
    fig.text(0.06, y, 'Per plane', fontsize=10, weight='bold')
    planes, flags, thresh = qc['planes'], qc['flags'], qc['thresholds']
    cols_x = [0.06] + [0.42 + i * 0.12 for i in range(len(planes))] + [0.42 + len(planes) * 0.12 + 0.02]
    y -= 0.03
    fig.add_artist(matplotlib.patches.Rectangle((0.06, y - 0.006), 0.88, 0.022, transform=fig.transFigure, facecolor=STYLE['panel'], edgecolor='none'))
    fig.text(cols_x[0] + 0.008, y, 'Metric', fontsize=8, weight='bold', color=STYLE['text_secondary'])
    for i, plane in enumerate(planes.index):
        fig.text(cols_x[i + 1] + 0.09, y, f'Plane {plane}', fontsize=8, weight='bold', color=STYLE['text_secondary'], ha='right')
    fig.text(cols_x[-1], y, 'Limit', fontsize=8, weight='bold', color=STYLE['text_secondary'])
    y -= 0.024
    for key, label, fmt, _ in METRICS:
        fig.text(cols_x[0] + 0.008, y, label, fontsize=8)
        for i, plane in enumerate(planes.index):
            v = planes.loc[plane, key]
            text = fmt.format(v) if np.isfinite(v) else '—'
            fig.text(cols_x[i + 1] + 0.09, y, text, fontsize=8, ha='right', weight='bold' if flags.loc[plane, key] else 'normal')
            if flags.loc[plane, key]:
                fig.text(cols_x[i + 1] + 0.012, y, WARN_ICON, fontsize=8.5, color=STYLE['warning'], weight='bold')
        if key in LIMITS:
            fig.text(cols_x[-1], y, LIMITS[key](thresh), fontsize=7.5, color=STYLE['text_muted'])
        fig.add_artist(matplotlib.lines.Line2D([0.06, 0.94], [y - 0.007, y - 0.007], transform=fig.transFigure, color=STYLE['grid'], lw=0.5))
        y -= 0.021

    # Warnings
    y -= 0.022
    n = len(qc['warnings'])
    fig.text(0.06, y, 'Warnings', fontsize=10, weight='bold')
    y -= 0.026
    if not n:
        fig.text(0.06, y, f'{OK_ICON}  All checks passed.', fontsize=8.5, color=STYLE['text'])
    for warning in qc['warnings']:
        lines = textwrap.wrap(warning, 110)
        fig.text(0.06, y, WARN_ICON, fontsize=9, color=STYLE['warning'], weight='bold')
        for line in lines:
            fig.text(0.085, y, line, fontsize=8)
            y -= 0.017
        y -= 0.006
        if y < 0.05:
            fig.text(0.085, y, '(more warnings not shown)', fontsize=8, color=STYLE['text_secondary'])
            break
    if qc.get('notes') and y > 0.1:
        y -= 0.016
        fig.text(0.06, y, 'Notes', fontsize=10, weight='bold')
        y -= 0.026
        for note in qc['notes']:
            fig.text(0.062, y, '•', fontsize=10, color=STYLE['text_secondary'], weight='bold')
            for line in textwrap.wrap(note, 110):
                fig.text(0.085, y, line, fontsize=8)
                y -= 0.017
            y -= 0.006
    pdf.savefig(fig)
    plt.close(fig)

def _trend_label(ax, change, flagged):
    """Start-to-end change in the corner of a panel, with an icon when it crosses a limit."""
    text = f'{change:+.0%} start → end'
    if flagged:
        text = f'{WARN_ICON} ' + text
    ax.text(0.99, 0.95, text, transform=ax.transAxes, ha='right', va='top', fontsize=7.5,
            color=STYLE['text'], weight='bold' if flagged else 'normal',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=STYLE['surface'], edgecolor='none', alpha=0.85))

def _registration_page(pdf, qc, npages):
    traces, planes, flags = qc['traces'], qc['planes'], qc['flags']
    nplanes = len(traces)
    fig = plt.figure(figsize=PAGE)
    _header(fig, qc, 'Registration and drift', 2, npages)
    fig.text(0.06, 0.905, f"Thin lines: every volume.  Thick lines: {qc['thresholds']['drift_window_s']} s running median.  "
                          'Registration quality is suite2p corrXY, the correlation of each frame with the reference image.',
             fontsize=7.5, color=STYLE['text_secondary'], wrap=True)
    gs = fig.add_gridspec(nplanes, 3, left=0.08, right=0.97, top=0.86, bottom=0.07, hspace=0.3, wspace=0.32)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(nplanes)])
    for plane, tr in traces.items():
        tmin = tr['t'] / 60
        # Rigid shift: x and y
        ax = axes[plane, 0]
        ax.plot(tmin, tr['xoff'], lw=0.3, color=STYLE['series_1_light'], rasterized=True)
        ax.plot(tmin, tr['yoff'], lw=0.3, color=STYLE['series_2_light'], rasterized=True)
        ax.plot(tmin, tr['xs'], lw=2, color=STYLE['series_1'], label='x')
        ax.plot(tmin, tr['ys'], lw=2, color=STYLE['series_2'], label='y')
        ax.set_ylabel(f'Plane {plane}', fontsize=9, weight='bold', color=STYLE['text'])
        ax.set_title('Rigid shift (px)' if plane == 0 else '', fontsize=8.5)
        ax.text(0.99, 0.95, f"drift {planes.loc[plane, 'lateral_drift_px']:.1f} px", transform=ax.transAxes, ha='right', va='top',
                fontsize=7.5, color=STYLE['text'], bbox=dict(boxstyle='round,pad=0.25', facecolor=STYLE['surface'], edgecolor='none', alpha=0.85))
        lim = max(3, np.abs(np.concatenate([tr['xoff'], tr['yoff']])).max() + 0.5)
        ax.set_ylim(-lim, lim)
        if plane == 0:
            ax.legend(loc='lower left', fontsize=7.5, ncols=2, handlelength=1.2, borderaxespad=0.2)
        # Registration quality
        ax = axes[plane, 1]
        ax.plot(tmin, tr['corr'], lw=0.3, color=STYLE['series_1_light'], rasterized=True)
        ax.plot(tmin, tr['cs'], lw=2, color=STYLE['series_1'])
        ax.set_title('Registration quality (corrXY)' if plane == 0 else '', fontsize=8.5)
        ax.set_ylim(np.percentile(tr['corr'], 0.5) * 0.95, np.percentile(tr['corr'], 99.5) * 1.05)
        _trend_label(ax, planes.loc[plane, 'corr_change'], flags.loc[plane, 'corr_change'])
        # Cell brightness
        ax = axes[plane, 2]
        ax.plot(tmin, tr['brightness_raw'], lw=0.3, color=STYLE['series_1_light'], rasterized=True)
        ax.plot(tmin, tr['brightness'], lw=2, color=STYLE['series_1'])
        ax.axhline(1, color=STYLE['text_muted'], lw=0.6)
        ax.set_title('Cell brightness (F / cell median)' if plane == 0 else '', fontsize=8.5)
        lo, hi = np.percentile(tr['brightness_raw'], [0.5, 99.5])
        ax.set_ylim(min(lo, 0.9), max(hi, 1.1))
        _trend_label(ax, planes.loc[plane, 'brightness_change'], flags.loc[plane, 'brightness_change'])
    for ax in axes.flat:
        ax.margins(x=0)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel('Time (min)', fontsize=8)
    pdf.savefig(fig, dpi=200)
    plt.close(fig)

def _images_page(pdf, qc, npages):
    traces, planes = qc['traces'], qc['planes']
    nplanes = len(traces)
    channels = [('meanImg', 'G (channel 1)')]
    if any(tr['meanImg_chan2'] is not None for tr in traces.values()):
        channels.append(('meanImg_chan2', 'R (channel 2)'))
    fig = plt.figure(figsize=PAGE)
    _header(fig, qc, 'Mean images', 3, npages)
    fig.text(0.06, 0.898, 'Registered mean image of each plane and channel. Each channel uses one scale for all planes,\n'
             'from the 1st to the 99.9th percentile of its pixels across planes.', fontsize=7.5, color=STYLE['text_secondary'])
    lims = {key: _limits([tr[key] for tr in traces.values()]) for key, _ in channels}
    gs = fig.add_gridspec(nplanes, len(channels), left=0.1, right=0.9, top=0.87, bottom=0.05, hspace=0.12, wspace=0.05)
    for plane, tr in traces.items():
        for icol, (key, name) in enumerate(channels):
            ax = fig.add_subplot(gs[plane, icol])
            img = tr[key]
            if img is not None:
                ax.imshow(img, cmap='gray', vmin=lims[key][0], vmax=lims[key][1], interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if plane == 0:
                ax.set_title(name, fontsize=9)
            if icol == 0:
                ax.set_ylabel(f"Plane {plane}\n{planes.loc[plane, 'selected_cells']:,} cells", fontsize=8.5, color=STYLE['text'])
    pdf.savefig(fig, dpi=200)
    plt.close(fig)

def _limits(imgs, lo=1, hi=99.9):
    """Display range shared by a set of images (e.g. all planes of one channel): percentiles of all their pixels."""
    vals = np.concatenate([np.ravel(img) for img in imgs if img is not None])
    vals = vals[np.isfinite(vals)]
    return (np.percentile(vals, lo), np.percentile(vals, hi)) if len(vals) else (0, 1)

def _show(ax, img, title=None, ylabel=None, lims=None):
    if img is not None and np.isfinite(img).any():
        vmin, vmax = lims if lims is not None else _limits([img])
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=STYLE['text'])

def _filters_page(pdf, qc, page, npages):
    """Session green image next to each filter* stack's registered mean images, with alignment per plane."""
    filters = qc['filters']
    columns = [('Session G', None, 'G')] + [(f'{name} {ch}', name, ch) for name, f in filters.items() for ch in f['channels']
                                             if not (f.get('scale', 1) > 1 and ch != 'G')]  # high-res stacks: green only
    nplanes = len(qc['session_g'])
    fig = plt.figure(figsize=PAGE)
    _header(fig, qc, 'Filter stacks', page, npages)
    fig.text(0.06, 0.898, 'Registered mean images; each column uses one scale for all planes (1st to 99.9th percentile).\n'
             'Stacks with the session\'s frame size are shifted onto the session; high-resolution stacks are shown as acquired.',
             fontsize=7.5, color=STYLE['text_secondary'])
    gs = fig.add_gridspec(nplanes, len(columns), left=0.08, right=0.97, top=0.86, bottom=0.30, hspace=0.12, wspace=0.04)
    get = lambda plane, name, ch: qc['session_g'][plane] if name is None else filters[name]['images'].get((plane, ch))
    for icol, (title, name, ch) in enumerate(columns):
        lims = _limits([get(plane, name, ch) for plane in range(nplanes)])
        for plane in range(nplanes):
            _show(fig.add_subplot(gs[plane, icol]), get(plane, name, ch), title if plane == 0 else None, f'Plane {plane}' if icol == 0 else None, lims)

    # Alignment table
    y = 0.255
    fig.text(0.06, y, 'Alignment to the session (green channel, rigid)', fontsize=10, weight='bold')
    y -= 0.028
    cols = [0.06, 0.24, 0.36, 0.52, 0.68, 0.82]
    for x, label in zip(cols, ['Stack', 'Plane', 'Shift (y, x)', 'Match before', 'Match after', 'z offset']):
        fig.text(x, y, label, fontsize=8, weight='bold', color=STYLE['text_secondary'])
    y -= 0.02
    for name, f in filters.items():
        for plane, shift in enumerate(f['shift_yx']):
            low = not np.isfinite(f['corr_after'][plane]) or f['corr_after'][plane] < qc['thresholds']['min_filter_corr']
            unit = 'px' if f.get('scale', 1) == 1 else f"px (at {qc['info']['Frame size'].split(' ')[0]} px)"
            values = [name if plane == 0 else '', str(plane), f'{shift[0]:+d}, {shift[1]:+d} {unit}', f"{f['corr_before'][plane]:.2f}",
                      f"{f['corr_after'][plane]:.2f}", f"{f['z_offset_um'][plane]:+g} µm" if plane < len(f['z_offset_um']) else '—']
            for x, v in zip(cols, values):
                fig.text(x, y, v, fontsize=8, weight='bold' if (low and x == cols[4]) else 'normal')
            if low:
                fig.text(cols[4] - 0.022, y, WARN_ICON, fontsize=8.5, color=STYLE['warning'], weight='bold')
            y -= 0.017
        y -= 0.006
    fig.text(0.06, max(y, 0.02), f"Match: correlation of the high-passed green images (limit {qc['thresholds']['min_filter_corr']}).",
             fontsize=7, color=STYLE['text_muted'])
    pdf.savefig(fig, dpi=200)
    plt.close(fig)

def write_report(qc, filename):
    npages = 4 if qc.get('filters') else 3
    with matplotlib.rc_context(RC):
        with PdfPages(filename) as pdf:
            _summary_page(pdf, qc, npages)
            _registration_page(pdf, qc, npages)
            _images_page(pdf, qc, npages)
            if qc.get('filters'):
                _filters_page(pdf, qc, 4, npages)

def main(mouse, date, session='session_1', ops=None):
    """
    Write qc_report.pdf for a session that has been through the suite2p and anndata steps.
    """
    if ops is None:
        ops = options.default_ops()
    adata = sess.load_as_anndata(mouse=mouse, date=date, session=session, ops=ops)
    if adata is None:
        raise FileNotFoundError(f'No adata.h5ad for {mouse} {date} {session}; run the anndata step first.')
    qc = compute_qc(adata, ops)
    filename = str(sess.define_path(mouse, date, session, ops=ops)['preprocessed_dir'] / 'qc_report.pdf')
    write_report(qc, filename)
    print(f'QC report saved to {filename}')
    # Machine-readable summary for the pipeline dashboard
    summary = {'warnings': [{'label': label, 'message': message} for label, message in zip(qc['warning_labels'], qc['warnings'])],
               'n_cells': qc['n_cells'], 'n_red': qc['n_red'], 'behavior_min': round(qc['behavior_min'], 2),
               'imaging_min': round(qc['imaging_min'], 2), 'notes': qc['notes'], 'created': time.strftime('%Y-%m-%d %H:%M')}
    with open(filename.replace('qc_report.pdf', 'qc_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1)
    for warning in qc['warnings']:
        print('QC warning:', warning)
    return qc
