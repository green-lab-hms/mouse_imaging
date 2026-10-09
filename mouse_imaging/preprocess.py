import copy, json, time, logging
from pathlib import Path
import numpy as np
import suite2p
from suite2p.run_s2p import logger_setup
from mouse_imaging import *
from mouse_imaging import compress

STEPS = ['suite2p', 'filters', 'anndata', 'qc', 'compress']

def suite2p_settings(overrides):
    """
    Merge nested overrides from ops['suite2p_settings'] into suite2p's default settings.
    """
    import torch
    def update(settings, overrides):
        for key, val in overrides.items():
            if isinstance(val, dict):
                update(settings[key], val)
            else:
                settings[key] = val
    settings = suite2p.default_settings()
    update(settings, overrides)
    settings['torch_device'] = 'cuda' if torch.cuda.is_available() else 'cpu' # use GPU if available for faster processing
    return settings

def _run_s2p(data_dir, suite2p_dir, md, ops, registration_only=False):
    """Run suite2p on the tifs in data_dir (all planes at once), writing to suite2p_dir."""
    tif_files = list(Path(data_dir).glob('*.tif'))
    assert len(tif_files) > 0, f'No tifs in {data_dir}'

    db = {
        'data_path': [str(data_dir)], # Directory where your input files are located
        'save_path0': str(Path(suite2p_dir).parent), # suite2p writes to save_path0/save_folder, so outputs land directly in suite2p_dir
        'save_folder': Path(suite2p_dir).name,
        # 'file_list': tif_files, # Specify files you'd like to specifically use in the data_path
        'input_format': 'tif',
        'nplanes': md['nslices'] + md['nflyback'], # each tiff has these many planes in sequence, including flyback frames
        'nchannels': md['nchannels'], # each tiff has these many channels per plane
        'keep_movie_raw': False,
        'batch_size': 200, # we will decrease the batch_size in case low RAM on computer
        'functional_chan': md['filter1'].get('functional_chan', 1), # the green channel; suite2p's meanImg is this one, meanImg_chan2 the other
        'ignore_flyback': list(range(md['nslices'], md['nslices'] + md['nflyback'])), # 0-based plane indices of flyback frames
    }

    settings = suite2p_settings(ops['suite2p_settings'])
    settings['fs'] = md['volume_rate'] # sampling rate of recording, determines binning for cell detection
    if registration_only: # motion correction and mean images only
        settings['run'].update(do_detection=False, do_deconvolution=False, do_regmetrics=False)

    logger_setup(suite2p_dir)
    try:
        suite2p.run_s2p(settings=settings, db=db)
    finally:
        # logger_setup adds a file handler on every call; remove it so later runs don't also write to this run.log
        s2p_logger = logging.getLogger('suite2p')
        for handler in [h for h in s2p_logger.handlers if isinstance(h, logging.FileHandler)]:
            s2p_logger.removeHandler(handler)
            handler.close()

def run_suite2p(path, md, ops):
    # Keep the registered binaries until the preview movie is made, then delete them as suite2p would
    ops = copy.deepcopy(ops)
    delete_bin = ops['suite2p_settings'].setdefault('io', {}).get('delete_bin', True)
    ops['suite2p_settings']['io']['delete_bin'] = False
    try:
        _run_s2p(path['twophoton_dir'], path['suite2p_dir'], md, ops)
        try:
            make_movie(path['suite2p_dir'], md, path['movie_mp4'])
        except Exception:
            import traceback
            traceback.print_exc()
            print('Preview movie failed; continuing without it.')
    finally:
        if delete_bin:
            for plane_dir in Path(path['suite2p_dir']).glob('plane*'):
                for name in ['data.bin', 'data_chan2.bin']:
                    if (plane_dir / name).is_file():
                        (plane_dir / name).unlink()

def make_movie(suite2p_dir, md, out_file, target_s=60, fps=30, tile=256, crf=28):
    """
    Sped-up preview of the registered recording (functional channel) as a small H.264 MP4, playable in any browser.
    Planes are tiled in a grid, each downsampled to about tile x tile px. Frames are averaged in bins so the whole
    recording plays in about target_s seconds at fps frames/s. Reads suite2p's registered plane*/data.bin.
    Encoded with the ffmpeg bundled in imageio-ffmpeg; crf sets quality vs file size (higher = smaller).
    """
    import math, subprocess, cv2, imageio_ffmpeg
    planes = []
    for plane in range(md['nslices']):
        ops_plane = np.load(Path(suite2p_dir) / f'plane{plane}' / 'ops.npy', allow_pickle=True).item()
        bin_file = Path(suite2p_dir) / f'plane{plane}' / 'data.bin'
        nframes, Ly, Lx = int(ops_plane['nframes']), int(ops_plane['Ly']), int(ops_plane['Lx'])
        planes.append(np.memmap(bin_file, dtype=np.int16, mode='r', shape=(nframes, Ly, Lx)))
    nframes = min(len(m) for m in planes)
    Ly, Lx = planes[0].shape[1:]
    nbin = max(1, math.ceil(nframes / (fps * target_s)))
    n_out = nframes // nbin
    down = max(1, round(Ly / tile))
    ty, tx = Ly // down, Lx // down
    ncols = min(len(planes), 3)
    nrows = math.ceil(len(planes) / ncols)
    H, W = nrows * ty + 24, ncols * tx  # 24 px strip for the time stamp
    H, W = H + H % 2, W + W % 2
    speed = nbin * fps / md['volume_rate']

    # Contrast per plane from a sample of binned frames
    sample = np.linspace(0, n_out - 1, min(n_out, 50)).astype(int)
    lims = []
    for m in planes:
        vals = np.stack([m[i * nbin:(i + 1) * nbin].mean(0)[::down, ::down] for i in sample])
        lims.append((np.percentile(vals, 1), np.percentile(vals, 99.7)))

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(out_file).with_suffix('.tmp.mp4')
    ffmpeg = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
                               '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', str(crf), '-pix_fmt', 'yuv420p',
                               '-movflags', '+faststart', str(tmp)], stdin=subprocess.PIPE)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(n_out):
        frame = np.zeros((H, W), np.uint8)
        for p, m in enumerate(planes):
            img = m[i * nbin:(i + 1) * nbin].mean(0)
            img = img[:ty * down, :tx * down].reshape(ty, down, tx, down).mean(axis=(1, 3))
            lo, hi = lims[p]
            r, c = divmod(p, ncols)
            frame[r * ty:(r + 1) * ty, c * tx:(c + 1) * tx] = np.clip((img - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
            cv2.putText(frame, f'plane {p}', (c * tx + 6, r * ty + 16), font, 0.4, 255, 1, cv2.LINE_AA)
        t = i * nbin / md['volume_rate']
        cv2.putText(frame, f'{int(t // 60):02d}:{int(t % 60):02d}   {speed:.0f}x speed', (6, H - 7), font, 0.45, 255, 1, cv2.LINE_AA)
        ffmpeg.stdin.write(frame.tobytes())
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError(f'ffmpeg failed with exit code {ffmpeg.returncode}')
    tmp.replace(out_file)
    print(f'Preview movie saved to {out_file} ({n_out} frames, {speed:.0f}x speed, {Path(out_file).stat().st_size / 1e6:.1f} MB)')

def run_filters(path, ops):
    """
    Motion-correct each filter* stack of the session (e.g. filter2: blue and green; filter1_1024: the session's filters at
    1024 x 1024) and save its time-averaged registered images in <preprocessed_dir>/<name>/:
        mean.tif       float32 ImageJ hyperstack, planes x channels x Ly x Lx, channels in ScanImage order (B, G, R)
        metadata.json  channels, frame size, planes, z positions, laser wavelength, number of volumes
        suite2p/       suite2p registration output (plane*/ops.npy with the shifts and reference image)
    Registration aligns on the green channel. The anndata step aligns these images to the session and measures each cell.
    """
    import tifffile, traceback
    if not path.get('filter_raw_dirs'):
        print('No filter* subfolders with tifs; nothing to do in the filters step.')
        return
    failed = []
    for name, raw_dir in path['filter_raw_dirs'].items():
        try:
            _run_filter(name, raw_dir, path['filter_dirs'][name], ops, tifffile)
        except Exception:
            # One bad stack shouldn't stop the session; the anndata step skips it and the QC report flags it
            traceback.print_exc()
            print(f'{name}: FAILED, skipping it (see the error above).')
            failed.append(name)
    if failed:
        print(f'filters step: {len(failed)} stack(s) failed: {", ".join(failed)}')

def _run_filter(name, raw_dir, out_dir, ops, tifffile):
    t0 = time.perf_counter()
    md = sess.read_si_tif_metadata(sorted(Path(raw_dir).glob('*.tif'))[0])
    channels = md['filter1']['channels']
    if len(channels) != md['nchannels']:
        raise ValueError(f"filename gives channels {channels} but ScanImage saved {md['nchannels']}; rename the files with a filter code, e.g. 850nm_G1B2_...")
    reg = 'G' if 'G' in channels else channels[0]
    print(f'{name}: {md["nslices"]} planes, channels {channels}, {md["Ly"]} x {md["Lx"]}, registering on {reg}')
    _run_s2p(raw_dir, out_dir / 'suite2p', md, ops, registration_only=True)

    # Mean images in channel order; suite2p's meanImg is the functional (green) channel, meanImg_chan2 the other one
    func = md['filter1'].get('functional_chan', 1) - 1
    other = [c for c in range(md['nchannels']) if c != func]
    mean = np.zeros((md['nslices'], md['nchannels'], md['Ly'], md['Lx']), dtype=np.float32)
    n_volumes = []
    for plane in range(md['nslices']):
        ops_plane = np.load(out_dir / 'suite2p' / f'plane{plane}' / 'ops.npy', allow_pickle=True).item()
        mean[plane, func] = ops_plane['meanImg']
        if other:
            mean[plane, other[0]] = ops_plane['meanImg_chan2']
        n_volumes.append(int(ops_plane['nframes']))
    tifffile.imwrite(out_dir / 'mean.tif', mean, imagej=True, metadata={'axes': 'ZCYX', 'Labels': [f'plane{p} {c}' for p in range(md['nslices']) for c in channels]})
    (out_dir / 'metadata.json').write_text(json.dumps({
        'name': name, 'source': str(raw_dir), 'channels': channels, 'nslices': md['nslices'], 'nflyback': md['nflyback'],
        'Ly': md['Ly'], 'Lx': md['Lx'], 'zs': md['zs'], 'zoom': md.get('SI.hRoiManager.scanZoomFactor'),
        'laser_nm': md['filter1'].get('laserWavelength_nm'), 'volume_rate': md['volume_rate'], 'n_volumes': n_volumes,
        'created': time.strftime('%Y-%m-%d %H:%M')}, indent=1))
    print(f'{name}: saved {out_dir / "mean.tif"} ({time.perf_counter() - t0:.0f} s)')

def main(mouse, date, session='session_1', ops_name='default_ops', steps=STEPS):
    ops = getattr(options, ops_name)()
    path = sess.define_path(mouse, date, session, ops=ops)
    md = sess.get_metadata(path)

    # Steps that read raw TIFFs need them uncompressed; the compress step compresses them again at the end
    if 'suite2p' in steps:
        compress.decompress_dir(path['twophoton_dir'])
    if 'filters' in steps:
        for raw_dir in path['filter_raw_dirs'].values():
            compress.decompress_dir(raw_dir)

    # Step 1: motion correction, ROI detection, extraction and deconvolution
    if 'suite2p' in steps:
        run_suite2p(path, md, ops)

    # Step 1b: motion-correct the filter* stacks (e.g. blue/green, high resolution) and save their mean images
    if 'filters' in steps:
        run_filters(path, ops)

    # Step 2: align suite2p output with sync and virmen data, and save as anndata
    if 'anndata' in steps:
        sess.main(mouse, date, session, ops=ops)

    # Step 3: QC report (qc_report.pdf): session summary, cell counts, registration and drift warnings
    if 'qc' in steps:
        from mouse_imaging import qc
        qc.main(mouse, date, session, ops=ops)

    # Step 4: losslessly compress the raw TIFFs (session folder and subfolders) to .tif.zst, about 57% of their size
    if 'compress' in steps:
        compress.compress_dir(path['twophoton_dir'])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess session.')
    parser.add_argument('--mouse', required=True, type=str)
    parser.add_argument('--date', required=True, type=str)
    parser.add_argument('--session', required=False, type=str, default='session_1')
    parser.add_argument('--ops', required=False, type=str, default='default_ops')
    parser.add_argument('--steps', required=False, type=str, default=','.join(STEPS),
                        help=f"Comma-separated steps to run, from: {','.join(STEPS)}")
    args = parser.parse_args()
    steps = args.steps.split(',')
    assert set(steps) <= set(STEPS), f'Unknown step in {steps}'
    main(args.mouse, args.date, args.session, ops_name=args.ops, steps=steps)
