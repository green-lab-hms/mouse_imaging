"""
Automatic preprocessing of newly uploaded sessions.

`scan()` (run every few minutes by cron, see tools/pipeline_cron.sh) finds every session folder under
<raw_root>/twophoton/<mouse>/<date>/session_N, checks its files, and submits preprocess.slurm for sessions
that are ready. Status is kept in <derived_root>/.pipeline/state.json, which the dashboard (dashboard.py) reads.

A session is ready when:
    - all three file types are present: ScanImage TIFFs, the ViRMEn sessionData.mat and the sync file
    - the upload has finished: nothing changed for `quiet_minutes`, the file list and sizes are the same as at
      the previous scan, TIFF file numbers are contiguous, all TIFFs but the last have the same size, and the
      last TIFF can be read
Ready sessions recorded on or after `auto_start_date` are submitted automatically; earlier ones wait for a
request from the dashboard (or `python -m mouse_imaging.pipeline queue <mouse> <date> <session>`).
A session missing its ViRMEn file can be requested with force=True ("Process anyway").

Usage:
    python -m mouse_imaging.pipeline scan [--no-submit]      # check sessions, submit ready ones
    python -m mouse_imaging.pipeline status                  # print the status table
    python -m mouse_imaging.pipeline queue JG6 260929 session_2 [--force]
"""
import os, re, json, time, fcntl, shutil, datetime, subprocess, contextlib
from pathlib import Path

from mouse_imaging import config

SESSION_RE = re.compile(r'^session_(\d+)$')
TIF_INDEX_RE = re.compile(r'_(\d+)\.tif$', re.IGNORECASE)
ACTIVE_JOB_STATES = {'PENDING', 'RUNNING', 'REQUEUED', 'RESIZING', 'SUSPENDED', 'CONFIGURING', 'COMPLETING'}
SLURM_SCRIPT = Path(__file__).resolve().parent / 'preprocess.slurm'

# Status values, in the order the dashboard lists them
STATUSES = ['failed', 'tiff_problem', 'running', 'pending', 'queued', 'missing_files', 'uploading', 'not_queued', 'done']

def settings():
    """[pipeline] settings from the config file, with defaults."""
    cfg = config.load_config().get('pipeline', {})
    derived_root = config.get_path('derived_root')
    return {
        'state_dir': Path(cfg.get('state_dir', derived_root / '.pipeline')).expanduser(),
        'quiet_minutes': float(cfg.get('quiet_minutes', 30)),
        'max_concurrent_jobs': int(cfg.get('max_concurrent_jobs', 2)),
        'auto_start_date': cfg.get('auto_start_date'), # None: the date of the first scan, saved in the state file
    }

# ---- State file ----

@contextlib.contextmanager
def locked_state(write=True):
    """Load state.json under an exclusive lock, and save it on exit if write=True."""
    s = settings()
    s['state_dir'].mkdir(parents=True, exist_ok=True)
    state_file = s['state_dir'] / 'state.json'
    with open(s['state_dir'] / 'state.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            state = json.loads(state_file.read_text()) if state_file.exists() else {}
        except json.JSONDecodeError:
            state = {}
        state.setdefault('sessions', {})
        state.setdefault('auto_start_date', s['auto_start_date'] or datetime.date.today().strftime('%y%m%d'))
        if s['auto_start_date']:
            state['auto_start_date'] = s['auto_start_date']
        yield state
        if write:
            tmp = state_file.with_suffix('.tmp')
            tmp.write_text(json.dumps(state, indent=1, default=str))
            os.replace(tmp, state_file)

def read_state():
    with locked_state(write=False) as state:
        return state

def format_date(date):
    """'261009' -> '2026-10-09'"""
    return f'20{date[:2]}-{date[2:4]}-{date[4:6]}' if re.match(r'^\d{6}$', str(date)) else str(date)

def session_id(mouse, date, session):
    return f'{mouse}/{date}/{session}'

# ---- Files ----

def discover_sessions():
    """(mouse, date, session) for every session_N folder under <raw_root>/twophoton."""
    root = config.get_path('raw_root') / 'twophoton'
    keys = []
    for mouse_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for date_dir in sorted(p for p in mouse_dir.iterdir() if p.is_dir()):
            for session_dir in sorted(p for p in date_dir.iterdir() if p.is_dir() and SESSION_RE.match(p.name)):
                keys.append((mouse_dir.name, date_dir.name, session_dir.name))
    return keys

def _files(directory, pattern):
    try:
        return [(e.name, e.stat()) for e in os.scandir(directory) if e.is_file() and pattern(e.name)]
    except FileNotFoundError:
        return []

def check_files(mouse, date, session):
    """Presence, size and modification time of the session's TIFF, ViRMEn and sync files."""
    raw_root = config.get_path('raw_root')
    tifs = sorted(_files(raw_root / 'twophoton' / mouse / date / session, lambda n: n.lower().endswith(('.tif', '.tiff'))))
    virmen = _files(raw_root / 'virmen' / mouse / date / session, lambda n: n == 'sessionData.mat')
    session_number = int(SESSION_RE.match(session).group(1))
    sync = _files(raw_root / 'sync' / mouse / date, lambda n: n.startswith(f'session_{session_number:03d}.'))
    all_files = tifs + virmen + sync
    return {
        'n_tifs': len(tifs),
        'tif_bytes': sum(st.st_size for _, st in tifs),
        'tif_names': [name for name, _ in tifs],
        'tif_sizes': [st.st_size for _, st in tifs],
        'virmen': bool(virmen),
        'sync': sync[0][0] if sync else None,
        'last_modified': max((st.st_mtime for _, st in all_files), default=None),
        'snapshot': [len(tifs), sum(st.st_size for _, st in all_files), len(all_files)],
    }

def check_tifs(files, mouse, date, session):
    """Problems that suggest an incomplete TIFF upload, as a list of messages (empty if none)."""
    problems = []
    names, sizes = files['tif_names'], files['tif_sizes']
    indices = [int(m.group(1)) for m in (TIF_INDEX_RE.search(n) for n in names) if m]
    if len(indices) == len(names) and indices:
        missing = sorted(set(range(min(indices), max(indices) + 1)) - set(indices))
        if missing:
            problems.append(f'TIFF file numbers missing: {missing[:10]}' + (' ...' if len(missing) > 10 else ''))
    if len(sizes) > 2 and len(set(sizes[:-1])) > 1:
        problems.append('TIFFs other than the last have different sizes (a file may still be copying)')
    if sizes and len(sizes) > 1 and sizes[-1] > max(sizes[:-1]):
        problems.append('The last TIFF is larger than the others')
    if names and not problems:
        import tifffile
        last = config.get_path('raw_root') / 'twophoton' / mouse / date / session / names[-1]
        try:
            with tifffile.TiffFile(last) as tif:
                tif.pages[-1]  # reads through the page list to the end of the file
        except Exception as e:
            problems.append(f'The last TIFF ({names[-1]}) cannot be read: {type(e).__name__}')
    return problems

# ---- Outputs and jobs ----

def outputs(mouse, date, session):
    derived = config.get_path('derived_root') / mouse / date / session
    qc_summary = derived / 'qc_summary.json'
    qc = json.loads(qc_summary.read_text()) if qc_summary.exists() else None
    logs = sorted(derived.glob('preprocess_*.log'), key=lambda p: p.stat().st_mtime)
    return {
        'derived_dir': str(derived),
        'adata': (derived / 'adata.h5ad').exists(),
        'qc_report': (derived / 'qc_report.pdf').exists(),
        'qc': qc,
        'log': str(logs[-1]) if logs else None,
    }

def job_states(job_ids):
    """SLURM state of each job id, from sacct."""
    job_ids = [str(j) for j in job_ids if j]
    if not job_ids:
        return {}
    out = subprocess.run(['sacct', '-n', '-X', '-P', '-o', 'JobID,State,Elapsed', '-j', ','.join(job_ids)],
                         capture_output=True, text=True).stdout
    states = {}
    for line in out.strip().splitlines():
        job_id, state, elapsed = line.split('|')
        states[job_id] = {'state': state.split()[0], 'elapsed': elapsed}
    return states

def conda_base():
    if os.environ.get('CONDA_BASE'):
        return os.environ['CONDA_BASE']
    exe = os.environ.get('CONDA_EXE') or shutil.which('conda')
    return str(Path(exe).resolve().parent.parent) if exe else None

def submit(mouse, date, session, test_only=False):
    """Submit preprocess.slurm for one session; returns the job id."""
    derived = config.get_path('derived_root') / mouse / date / session
    derived.mkdir(parents=True, exist_ok=True)
    log = str(derived / 'preprocess_%j.log')
    cmd = ['sbatch', '--parsable', '-o', log, '-e', log, '-J', f'pre_{mouse}_{date}_{session}']
    base = conda_base()
    if base:
        cmd += ['--export', f'ALL,CONDA_BASE={base}'] # conda may not be on PATH when run from cron
    if test_only:
        cmd.append('--test-only')
    cmd += [str(SLURM_SCRIPT), mouse, date, session]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(derived))
    if result.returncode != 0:
        raise RuntimeError(f'sbatch failed: {result.stderr.strip()}')
    if test_only:
        return result.stderr.strip()
    return result.stdout.strip().split(';')[0]

# ---- Status ----

def evaluate(entry, files, out, now, auto_start_date, quiet_minutes):
    """Status of one session and a short explanation."""
    job = entry.get('job')
    if job:
        state = job.get('state', 'PENDING')
        if state in ACTIVE_JOB_STATES:
            return ('running' if state == 'RUNNING' else 'pending'), f"SLURM job {job['id']} {state.lower()}"
        if state == 'COMPLETED' and out['adata']:
            return 'done', f"Preprocessed by job {job['id']}"
        return 'failed', f"SLURM job {job['id']} {state.lower()}" + ('' if out['adata'] else ', no adata.h5ad')
    if entry.get('submit_error') and not entry.get('request'):
        return 'failed', entry['submit_error']
    if out['adata'] and not entry.get('request'):
        return 'done', 'Preprocessed'

    missing = [name for name, ok in [('TIFFs', files['n_tifs'] > 0), ('ViRMEn', files['virmen']), ('sync', files['sync'])] if not ok]
    force = entry.get('request', {}).get('force', False)
    if missing and not (force and missing == ['ViRMEn']):
        note = f". Also: {entry['tif_problems'][0]}" if entry.get('tif_problems') else ''
        return 'missing_files', 'Missing ' + ', '.join(missing) + note

    # Upload has finished when nothing changed for quiet_minutes and the snapshot matches the last scan
    quiet = files['last_modified'] is not None and (now - files['last_modified']) / 60 >= quiet_minutes
    stable = entry.get('snapshot') == files['snapshot']
    if not (quiet and stable):
        minutes = (now - files['last_modified']) / 60 if files['last_modified'] else 0
        return 'uploading', f'Last file change {minutes:.0f} min ago'
    if entry.get('tif_problems') and not force:
        return 'tiff_problem', entry['tif_problems'][0] + '. Check the upload, or process anyway.'

    if entry.get('request'):
        return 'queued', 'Requested from ' + entry['request'].get('by', 'dashboard') + (' (without ViRMEn)' if force else '')
    if entry['date'] >= auto_start_date:
        return 'queued', 'New session, queued automatically'
    return 'not_queued', f'Recorded before {format_date(auto_start_date)}, so not processed automatically'

def scan(submit_jobs=True, verbose=False):
    """Check every session, update state.json and submit ready sessions. Returns the state."""
    s = settings()
    now = time.time()
    with locked_state() as state:
        sessions = state['sessions']
        # Refresh job states
        states = job_states([e['job']['id'] for e in sessions.values() if e.get('job')])
        for entry in sessions.values():
            if entry.get('job') and entry['job']['id'] in states:
                entry['job'].update(states[entry['job']['id']])

        for mouse, date, session in discover_sessions():
            sid = session_id(mouse, date, session)
            entry = sessions.setdefault(sid, {'mouse': mouse, 'date': date, 'session': session, 'first_seen': now})
            files = check_files(mouse, date, session)
            out = outputs(mouse, date, session)
            # Read the last TIFF once per snapshot, when the upload looks finished
            if files['snapshot'] != entry.get('tif_checked_snapshot') and files['n_tifs'] and entry.get('snapshot') == files['snapshot']:
                entry['tif_problems'] = check_tifs(files, mouse, date, session)
                entry['tif_checked_snapshot'] = files['snapshot']
            status, detail = evaluate(entry, files, out, now, state['auto_start_date'], s['quiet_minutes'])
            entry.update(snapshot=files['snapshot'], status=status, detail=detail, checked=now,
                         files={k: files[k] for k in ['n_tifs', 'tif_bytes', 'virmen', 'sync', 'last_modified']},
                         outputs=out)

        if submit_jobs:
            _submit_queued(state, s['max_concurrent_jobs'], verbose)
        state['last_scan'] = now
        return state

def _submit_queued(state, max_jobs, verbose=False):
    sessions = state['sessions']
    active = sum(e['status'] in ('running', 'pending') for e in sessions.values())
    queued = sorted((e for e in sessions.values() if e['status'] == 'queued'), key=lambda e: (e.get('request', {}).get('time', 0) == 0, e['date']))
    for entry in queued:
        if active >= max_jobs:
            entry['detail'] += f' (waiting: {active} jobs running, limit {max_jobs})'
            continue
        try:
            job_id = submit(entry['mouse'], entry['date'], entry['session'])
        except RuntimeError as e:
            entry.update(status='failed', detail=str(e), submit_error=str(e))
            entry.pop('request', None)
            continue
        entry['job'] = {'id': job_id, 'state': 'PENDING', 'submitted': time.time()}
        entry.update(status='pending', detail=f'SLURM job {job_id} submitted')
        entry.pop('request', None)
        active += 1
        if verbose:
            print(f"Submitted {session_id(entry['mouse'], entry['date'], entry['session'])} as job {job_id}")

def request(mouse, date, session, force=False, by='dashboard'):
    """Queue a session for preprocessing (dashboard buttons). force=True allows a missing ViRMEn file.
    A failed or finished session is reprocessed."""
    with locked_state() as state:
        sid = session_id(mouse, date, session)
        entry = state['sessions'].get(sid)
        if entry is None:
            raise KeyError(f'Unknown session {sid}; it has not been scanned yet.')
        if entry.get('status') in ('running', 'pending'):
            raise ValueError(f'{sid} is already {entry["status"]}.')
        entry.pop('job', None)  # forget a previous failed or finished job
        entry.pop('submit_error', None)
        entry['request'] = {'time': time.time(), 'force': bool(force), 'by': by}
    # Rescan so the request is evaluated and submitted right away if a job slot is free
    return scan()['sessions'][sid]

def status_table(state=None):
    import pandas as pd
    state = state or read_state()
    rows = [{'session': sid, 'status': e.get('status'), 'detail': e.get('detail'), 'tifs': e.get('files', {}).get('n_tifs'),
             'virmen': e.get('files', {}).get('virmen'), 'sync': bool(e.get('files', {}).get('sync')),
             'qc_warnings': len((e.get('outputs', {}).get('qc') or {}).get('warnings', []))} for sid, e in sorted(state['sessions'].items())]
    return pd.DataFrame(rows)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Automatic preprocessing of new sessions.')
    sub = parser.add_subparsers(dest='command', required=True)
    p_scan = sub.add_parser('scan', help='check sessions and submit ready ones')
    p_scan.add_argument('--no-submit', action='store_true', help='only update the status')
    sub.add_parser('status', help='print the status of every session')
    p_queue = sub.add_parser('queue', help='queue one session for preprocessing')
    p_queue.add_argument('mouse'); p_queue.add_argument('date'); p_queue.add_argument('session')
    p_queue.add_argument('--force', action='store_true', help='process even if the ViRMEn file is missing')
    args = parser.parse_args()
    if args.command == 'scan':
        state = scan(submit_jobs=not args.no_submit, verbose=True)
        print(time.strftime('%Y-%m-%d %H:%M:%S'), 'scan done:', status_table(state)['status'].value_counts().to_dict())
    elif args.command == 'status':
        import pandas as pd
        with pd.option_context('display.width', 200, 'display.max_rows', None, 'display.max_colwidth', 60):
            print(status_table())
    elif args.command == 'queue':
        entry = request(args.mouse, args.date, args.session, force=args.force, by='command line')
        print(entry['status'], '-', entry['detail'])
