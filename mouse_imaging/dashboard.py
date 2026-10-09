"""
Web dashboard for the preprocessing pipeline (pipeline.py).

Shows every session's files, status, SLURM job and QC flags, with buttons to process sessions that aren't queued
automatically. Uses only the Python standard library.

    python -m mouse_imaging.dashboard [--port 8050]

Open the printed URL. Through a VS Code tunnel, VS Code forwards the port automatically (Ports panel).
To keep it running with a permanent web address, see tools/dashboard_service.sh and tools/PIPELINE.md.
The server only listens on this machine (127.0.0.1) and requires the access token in the URL, so other users on the
node can't submit jobs as you. The token is kept in ~/.config/mouse_imaging/dashboard_token.
"""
import json, re, secrets, mimetypes, time, html, datetime
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from mouse_imaging import pipeline, config

TOKEN_FILE = Path.home() / '.config' / 'mouse_imaging' / 'dashboard_token'
PART_RE = re.compile(r'^[A-Za-z0-9_.-]+$')

def get_token():
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)
    TOKEN_FILE.write_text(token)
    TOKEN_FILE.chmod(0o600)
    return token

def status_json():
    state = pipeline.read_state()
    now = time.time()
    # Refresh running and pending jobs live, so estimates don't wait for the next scan (not written to state.json)
    active = {e['job']['id']: e for e in state['sessions'].values() if e.get('status') in ('running', 'pending') and e.get('job')}
    for job_id, live in pipeline.job_states(list(active)).items():
        e = active[job_id]
        e['job'].update(live)
        if live['state'] == 'RUNNING':
            e['status'], e['detail'] = 'running', f'SLURM job {job_id} running'
        elif live['state'] not in pipeline.ACTIVE_JOB_STATES:
            e['detail'] = f'SLURM job {job_id} {live["state"].lower()}; updating at the next scan'
    sessions = []
    for sid, e in sorted(state['sessions'].items()):
        files, out = e.get('files', {}), e.get('outputs', {})
        qc = out.get('qc') or {}
        sessions.append({
            'id': sid, 'mouse': e['mouse'], 'date': e['date'], 'session': e['session'],
            'status': e.get('status'), 'detail': e.get('detail'), 'tif_notes': e.get('tif_notes') or [], 'ignored': bool(e.get('ignored')),
            'n_tifs': files.get('n_tifs', 0), 'empty_folder': files.get('empty_folder', False), 'n_compressed': files.get('n_compressed', 0), 'tif_gb': round(files.get('tif_bytes', 0) / 1e9, 1),
            'virmen': files.get('virmen', False), 'sync': bool(files.get('sync')), 'filter_stacks': files.get('filter_stacks', {}),
            'done_at': out.get('adata_time'),
            'job': e.get('job'), 'has_log': bool(out.get('log')), 'qc_report': out.get('qc_report', False), 'movie': out.get('movie', False),
            'qc_warnings': qc.get('warnings', []), 'n_cells': qc.get('n_cells'), 'n_red': qc.get('n_red'),
            'behavior_min': qc.get('behavior_min'),
            'eta': pipeline.estimate(e, state, now) if e.get('status') in ('running', 'pending', 'queued', 'not_queued') else None,
        })
    s = pipeline.settings()
    return {'last_scan': state.get('last_scan'), 'auto_start_date': state.get('auto_start_date'),
            'max_concurrent_jobs': s['max_concurrent_jobs'], 'quiet_minutes': s['quiet_minutes'], 'sessions': sessions}

def session_parts(path_parts):
    if len(path_parts) != 3 or not all(PART_RE.match(p) for p in path_parts) or not pipeline.SESSION_RE.match(path_parts[2]):
        raise ValueError('bad session path')
    return path_parts

SUBPAGE_CSS = """
:root { color-scheme: light; --surface: #fcfcfb; --panel: #f4f3f0; --line: #e6e5e1; --text: #0b0b0b; --text-2: #52514e;
  --text-3: #8a8985; --accent: #2a78d6; --good: #0ca30c; --critical: #d03b3b; --warning: #b77a00; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark; --surface: #1a1a19; --panel: #242422;
  --line: #383835; --text: #ffffff; --text-2: #c3c2b7; --text-3: #8f8e86; --accent: #3987e5; --good: #2fbf2f; --critical: #e66767; --warning: #fab219; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--text); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 24px 0 6px; }
.sub, .small { color: var(--text-2); font-size: 13px; } .muted { color: var(--text-3); }
code { font-size: 12px; overflow-wrap: anywhere; }
.box { border: 1px solid var(--line); border-radius: 8px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th { text-align: left; font-size: 12px; color: var(--text-2); background: var(--panel); padding: 6px 10px; font-weight: 600; }
td { padding: 5px 10px; border-top: 1px solid var(--line); font-size: 13px; white-space: nowrap; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.ok { color: var(--good); font-weight: 700; } .no { color: var(--critical); font-weight: 700; }
tr.expected td:first-child { font-weight: 600; }
.status { margin: 6px 0; font-size: 13px; }
video { width: 100%; max-height: 80vh; background: #000; border-radius: 8px; }
a { color: var(--accent); }
"""

def _subpage(title, body):
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{SUBPAGE_CSS}</style></head><body><main>{body}</main></body></html>')

def movie_page(mouse, date, session, movie):
    mb = movie.stat().st_size / 1e6
    made = datetime.datetime.fromtimestamp(movie.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
    src = f'/movie/{mouse}/{date}/{session}/movie.mp4'
    return _subpage(f'{mouse} {date} {session} movie', f"""
      <h1>{html.escape(mouse)} · {html.escape(date)} · {html.escape(session)}</h1>
      <div class="sub">Motion-corrected recording, sped up, functional channel, all planes. {mb:.1f} MB, made {made}.
        <a href="{src}" download>Download</a></div>
      <p><video src="{src}" controls autoplay loop muted playsinline></video></p>""")

def _fmt_size(n):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1000 or unit == 'TB':
            return f'{n:.0f} {unit}' if unit == 'B' else f'{n:.1f} {unit}'
        n /= 1000

def _listing(directory, expected=None, max_rows=20):
    """HTML table of a folder's files and subfolders. expected(name) marks the files the pipeline looks for."""
    if not directory.exists():
        parent = directory.parent
        siblings = sorted(e.name for e in parent.iterdir()) if parent.exists() else []
        hint = (f'<div class="small">{html.escape(str(parent))} contains: ' + (', '.join(f'<code>{html.escape(n)}</code>' for n in siblings[:40]) or '<span class="muted">nothing</span>')
                + (' …' if len(siblings) > 40 else '') + '</div>') if parent.exists() else f'<div class="small">{html.escape(str(parent))} doesn\'t exist either.</div>'
        return f'<div class="status"><span class="no">✕</span> This folder doesn\'t exist: <code>{html.escape(str(directory))}</code></div>{hint}'
    entries = []
    for e in sorted(directory.iterdir(), key=lambda e: (not e.is_dir(), e.name)):
        try:
            st = e.stat()
        except OSError:
            continue
        if e.is_dir():
            files = [f for f in e.iterdir() if f.is_file()]
            size, detail = sum(f.stat().st_size for f in files), f'folder, {len(files)} files'
        else:
            size, detail = st.st_size, ''
        entries.append((e.name, e.is_dir(), size, st.st_mtime, detail))
    if not entries:
        return f'<div class="status muted">The folder is empty: <code>{html.escape(str(directory))}</code></div>'
    rows = entries if len(entries) <= max_rows else entries[:max_rows // 2] + [None] + entries[-max_rows // 2:]
    out = []
    for r in rows:
        if r is None:
            out.append(f'<tr><td colspan="4" class="muted">… {len(entries) - max_rows} more …</td></tr>')
            continue
        name, is_dir, size, mtime, detail = r
        mark = expected is not None and expected(name)
        out.append(f'<tr class="{"expected" if mark else ""}"><td>{html.escape(name)}{"/" if is_dir else ""}</td><td class="num">{_fmt_size(size)}</td>'
                   f'<td>{datetime.datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M}</td><td class="small">{detail}</td></tr>')
    total = sum(e[2] for e in entries)
    return (f'<div class="small"><code>{html.escape(str(directory))}</code> · {len(entries)} items · {_fmt_size(total)}</div>'
            f'<div class="box"><table><thead><tr><th>Name</th><th>Size</th><th>Modified</th><th></th></tr></thead><tbody>{"".join(out)}</tbody></table></div>')

def files_page(mouse, date, session):
    """Where the pipeline looks for a session's files, what it found, and what each folder actually contains."""
    raw, derived = config.get_path('raw_root'), config.get_path('derived_root')
    files = pipeline.check_files(mouse, date, session)
    entry = pipeline.read_state()['sessions'].get(pipeline.session_id(mouse, date, session), {})
    n = int(pipeline.SESSION_RE.match(session).group(1))
    tif = lambda name: name.lower().endswith(('.tif', '.tiff', '.tif.zst', '.tiff.zst'))
    def check(ok, text):
        return f'<div class="status">{"<span class=ok>✓</span>" if ok else "<span class=no>✕</span>"} {text}</div>'
    two = raw / 'twophoton' / mouse / date / session
    sections = [
        ('ScanImage TIFFs', check(files['n_tifs'] > 0, f"{files['n_tifs']} TIFFs ({_fmt_size(files['tif_bytes'])}) directly in the session folder"
                                  + ''.join(f'; {k}: {v} TIFFs' for k, v in files.get('filter_stacks', {}).items()))
         + ''.join(f'<div class="status"><span class="no">!</span> {html.escape(p)}</div>' for p in entry.get('tif_problems') or []),
         two, tif),
        ('ViRMEn', check(files['virmen'], 'sessionData.mat'), raw / 'virmen' / mouse / date / session, lambda name: name == 'sessionData.mat'),
        ('Sync', check(bool(files['sync']), f'session_{n:03d}.* in the date folder' + (f" (found {html.escape(files['sync'])})" if files['sync'] else '')),
         raw / 'sync' / mouse / date, lambda name: name.startswith(f'session_{n:03d}.')),
        ('Derived (preprocessing output)', '', derived / mouse / date / session, lambda name: name in ('adata.h5ad', 'qc_report.pdf', 'suite2p')),
    ]
    body = [f'<h1>Files for {html.escape(mouse)} · {html.escape(date)} · {html.escape(session)}</h1>',
            f'<div class="sub">Status: <b>{html.escape(str(entry.get("status", "unknown")))}</b> — {html.escape(str(entry.get("detail", "")))}. '
            'Bold rows are the files the pipeline looks for.</div>']
    for title, summary, directory, expected in sections:
        body.append(f'<h2>{title}</h2>{summary}{_listing(directory, expected)}')
        if title == 'ScanImage TIFFs' and directory.exists():
            for sub in sorted(d for d in directory.iterdir() if d.is_dir() and d.name.startswith('filter')):
                body.append(f'<h2 class="small">{html.escape(sub.name)}</h2>{_listing(sub, tif)}')
    return _subpage(f'Files {mouse} {date} {session}', ''.join(body))

class Handler(BaseHTTPRequestHandler):
    token = None

    def log_message(self, fmt, *args):
        pass  # keep the terminal quiet

    def _authorized(self):
        query_token = parse_qs(urlparse(self.path).query).get('token', [''])[0]
        cookies = dict(c.strip().split('=', 1) for c in self.headers.get('Cookie', '').split(';') if '=' in c)
        return any(secrets.compare_digest(t.encode(), self.token.encode())
                   for t in (query_token, cookies.get('mi_token', '')) if t)

    def _send(self, code, body, content_type='application/json', extra_headers=()):
        data = body if isinstance(body, bytes) else (json.dumps(body) if content_type == 'application/json' else body).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        for key, val in extra_headers:
            self.send_header(key, val)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parts = [p for p in urlparse(self.path).path.split('/') if p]
        if not self._authorized():
            if not parts:
                return self._send(403, LOGIN_PAGE, 'text/html; charset=utf-8')
            return self._send(403, 'Forbidden: open the dashboard home page and enter the access token first.', 'text/plain')
        try:
            if not parts:
                cookie = ('Set-Cookie', f'mi_token={self.token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=31536000')
                return self._send(200, PAGE, 'text/html; charset=utf-8', [cookie])
            if parts == ['api', 'status']:
                return self._send(200, status_json())
            if parts == ['api', 'delete_preview']:
                mouse, date, session = session_parts(parse_qs(urlparse(self.path).query).get('id', [''])[0].split('/'))
                tifs, sync = pipeline.raw_files_to_delete(mouse, date, session)
                folders = sorted({str(f.parent) for f in tifs})
                return self._send(200, {'n_tifs': len(tifs), 'tif_gb': round(sum(f.stat().st_size for f in tifs) / 1e9, 1),
                                        'folders': folders, 'sync': [str(f) for f in sync]})
            if parts[0] in ('qc', 'log'):
                mouse, date, session = session_parts(parts[1:])
                out = pipeline.outputs(mouse, date, session)
                if parts[0] == 'qc':
                    pdf = Path(out['derived_dir']) / 'qc_report.pdf'
                    if not pdf.exists():
                        return self._send(404, 'No QC report yet.', 'text/plain')
                    return self._send(200, pdf.read_bytes(), 'application/pdf',
                                      [('Content-Disposition', f'inline; filename="qc_{mouse}_{date}_{session}.pdf"')])
                if not out['log']:
                    return self._send(404, 'No log yet.', 'text/plain')
                lines = Path(out['log']).read_text(errors='replace').splitlines()
                text = f"{out['log']}  (last {min(len(lines), 300)} of {len(lines)} lines)\n\n" + '\n'.join(lines[-300:])
                return self._send(200, text, 'text/plain; charset=utf-8')
            if parts[0] == 'movie' and len(parts) in (4, 5):
                mouse, date, session = session_parts(parts[1:4])
                movie = config.get_path('derived_root') / mouse / date / session / 'movie.mp4'
                if not movie.exists():
                    return self._send(404, 'No preview movie for this session. It is made when the suite2p step runs.', 'text/plain')
                if len(parts) == 5 and parts[4] == 'movie.mp4':
                    return self._send_file(movie, 'video/mp4')
                return self._send(200, movie_page(mouse, date, session, movie), 'text/html; charset=utf-8')
            if parts[0] == 'files':
                mouse, date, session = session_parts(parts[1:])
                return self._send(200, files_page(mouse, date, session), 'text/html; charset=utf-8')
        except ValueError:
            return self._send(400, 'Bad request', 'text/plain')
        self._send(404, 'Not found', 'text/plain')

    def _send_file(self, filename, content_type):
        """Send a file, honoring a single byte Range so video players can seek."""
        size = filename.stat().st_size
        start, end = 0, size - 1
        m = re.match(r'bytes=(\d*)-(\d*)$', self.headers.get('Range', ''))
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start, end = int(m.group(1)), min(int(m.group(2) or size - 1), size - 1)
            else:  # last N bytes
                start = max(0, size - int(m.group(2)))
            if start > end:
                self.send_response(416); self.send_header('Content-Range', f'bytes */{size}'); self.end_headers(); return
        with open(filename, 'rb') as fh:
            fh.seek(start)
            data = fh.read(end - start + 1)
        self.send_response(206 if m and (m.group(1) or m.group(2)) else 200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-store')
        if m and (m.group(1) or m.group(2)):
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if not self._authorized():
            return self._send(403, {'error': 'forbidden'})
        parts = [p for p in urlparse(self.path).path.split('/') if p]
        length = int(self.headers.get('Content-Length', 0) or 0)
        body = json.loads(self.rfile.read(length) or b'{}') if length else {}
        try:
            if parts == ['api', 'scan']:
                pipeline.scan()
                return self._send(200, status_json())
            if parts == ['api', 'queue']:
                mouse, date, session = session_parts(body.get('id', '').split('/'))
                entry = pipeline.request(mouse, date, session, force=bool(body.get('force')), by='dashboard')
                return self._send(200, {'status': entry['status'], 'detail': entry['detail']})
            if parts == ['api', 'delete']:
                sid = body.get('id', '')
                if body.get('confirm') != sid:
                    raise ValueError('Confirmation does not match the session name.')
                entry, n_tifs, n_sync, nbytes = pipeline.delete_raw(*session_parts(sid.split('/')), by='dashboard')
                return self._send(200, {'n_tifs': n_tifs, 'n_sync': n_sync, 'gb': round(nbytes / 1e9, 1), 'status': entry['status']})
            if parts == ['api', 'delete_folder']:
                folder = pipeline.delete_empty_folder(*session_parts(body.get('id', '').split('/')), by='dashboard')
                return self._send(200, {'folder': folder})
            if parts == ['api', 'ignore']:
                mouse, date, session = session_parts(body.get('id', '').split('/'))
                entry = pipeline.set_ignored(mouse, date, session, ignored=bool(body.get('ignored', True)), by='dashboard')
                return self._send(200, {'status': entry['status'], 'detail': entry['detail'], 'ignored': bool(entry.get('ignored'))})
        except (KeyError, ValueError, RuntimeError) as e:
            return self._send(400, {'error': str(e)})
        self._send(404, {'error': 'not found'})

def main(port=8050, url=None):
    Handler.token = get_token()
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'Dashboard running. Open:\n    {url or f"http://localhost:{port}"}/?token={Handler.token}\n'
          'In a VS Code tunnel, the port is forwarded automatically (see the Ports panel). Press Ctrl+C to stop.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

LOGIN_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pipeline dashboard</title>
<style>
:root { --bg: #f7f7f5; --card: #fff; --ink: #1f2328; --muted: #656d76; --line: #d8dee4; --accent: #2a78d6; }
@media (prefers-color-scheme: dark) { :root { --bg: #16181c; --card: #1f2227; --ink: #e6e8eb; --muted: #9aa3ad; --line: #353a42; --accent: #5b9be6; } }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 system-ui, sans-serif; display: grid; place-items: center; min-height: 100vh; }
form { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 24px; width: min(420px, calc(100vw - 32px)); box-sizing: border-box; }
h1 { font-size: 18px; margin: 0 0 6px; } p { color: var(--muted); margin: 0 0 16px; font-size: 13px; }
code { font-size: 12px; }
input { width: 100%; box-sizing: border-box; padding: 8px 10px; border: 1px solid var(--line); border-radius: 6px; background: var(--bg); color: var(--ink); font: inherit; }
button { margin-top: 12px; width: 100%; padding: 8px; border: 0; border-radius: 6px; background: var(--accent); color: #fff; font: inherit; font-weight: 600; cursor: pointer; }
</style></head><body>
<form method="get" action="/">
  <h1>Pipeline dashboard</h1>
  <p>Enter the access token. It's in <code>~/.config/mouse_imaging/dashboard_token</code> on spinoza. Your browser remembers it after this.</p>
  <input name="token" type="password" autocomplete="current-password" placeholder="Access token" autofocus required>
  <button type="submit">Open dashboard</button>
</form></body></html>
"""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Preprocessing Pipeline</title>
<style>
:root {
  color-scheme: light;
  --bg: #f5f5f2; --surface: #ffffff; --panel: #f3f3ef; --line: #e7e6e1; --line-2: #d9d8d2;
  --text: #121211; --text-2: #55544f; --text-3: #8c8b85;
  --accent: #2a78d6; --accent-ink: #ffffff; --accent-soft: #e8f1fc;
  --good: #11891a; --good-soft: #e6f4e7; --warning: #a86f00; --warning-soft: #fbf1db;
  --serious: #c4561f; --serious-soft: #fbebe2; --critical: #c93636; --critical-soft: #fbe8e8; --neutral-soft: #eeeeea;
  --shadow: 0 1px 2px rgba(20, 20, 15, .05), 0 1px 1px rgba(20, 20, 15, .03);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #141413; --surface: #1c1c1b; --panel: #242422; --line: #2f2f2c; --line-2: #3d3d39;
    --text: #f3f3f0; --text-2: #c0bfb7; --text-3: #8d8c85;
    --accent: #4a92e8; --accent-ink: #ffffff; --accent-soft: #1d2b3d;
    --good: #45c24d; --good-soft: #1b2d1c; --warning: #f0b232; --warning-soft: #33291a;
    --serious: #ec835a; --serious-soft: #3a2519; --critical: #ec6b6b; --critical-soft: #3a1f1f; --neutral-soft: #2a2a27;
    --shadow: 0 1px 2px rgba(0, 0, 0, .4);
  }
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; -webkit-font-smoothing: antialiased; }
main { max-width: 1320px; margin: 0 auto; padding: 28px 20px 56px; }
a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
button { font: inherit; font-size: 13px; font-weight: 500; border-radius: 7px; border: 1px solid var(--line-2); background: var(--surface); color: var(--text);
  padding: 5px 12px; cursor: pointer; white-space: nowrap; transition: background .12s, border-color .12s; }
button:hover { background: var(--panel); }
button.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
button.primary:hover { filter: brightness(1.06); }
button.act { border-color: var(--accent); color: var(--accent); }
button.act:hover { background: var(--accent-soft); }
button:disabled { opacity: .5; cursor: progress; }

/* Header */
header { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 16px; justify-content: space-between; margin-bottom: 18px; }
.brand { display: flex; align-items: center; gap: 12px; }
.mark { width: 36px; height: 36px; border-radius: 9px; background: var(--accent); color: var(--accent-ink); display: grid; place-items: center; font-weight: 700; font-size: 13px; letter-spacing: .02em; flex: none; }
h1 { font-size: 20px; margin: 0; letter-spacing: -.01em; }
.sub { color: var(--text-2); font-size: 12.5px; margin-top: 1px; }
.live { display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: var(--good); margin-right: 6px; vertical-align: 1px; box-shadow: 0 0 0 3px var(--good-soft); }
.headright { display: flex; align-items: center; gap: 12px; }

/* Tabs */
.tabs { display: flex; align-items: flex-end; gap: 4px; border-bottom: 1px solid var(--line); margin-bottom: 16px; }
.tab { background: none; border: 0; border-bottom: 2px solid transparent; border-radius: 0; padding: 7px 10px; color: var(--text-2); font-weight: 600; margin-bottom: -1px; }
.tab:hover { background: none; color: var(--text); }
.tab[aria-selected="true"] { color: var(--text); border-bottom-color: var(--accent); }
.tab.minor { margin-left: auto; font-size: 12px; font-weight: 500; }
.tab .count { display: inline-block; min-width: 20px; padding: 0 6px; margin-left: 6px; border-radius: 999px; background: var(--neutral-soft); color: var(--text-2); font-size: 11.5px; font-weight: 600; text-align: center; }

/* Status tiles */
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 16px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; text-align: left; box-shadow: var(--shadow); display: grid; grid-template-columns: auto 1fr; column-gap: 10px; align-items: center; }
.tile:hover { background: var(--surface); border-color: var(--line-2); }
.tile[aria-pressed="true"] { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent); }
.tile .ic { grid-row: span 2; width: 30px; height: 30px; border-radius: 8px; display: grid; place-items: center; font-size: 14px; font-weight: 700; }
.tile .n { font-size: 21px; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
.tile .l { font-size: 12px; color: var(--text-2); }

/* Status colors: icon and soft background per state */
.c-all { background: var(--neutral-soft); color: var(--text-2); }
.c-done { background: var(--good-soft); color: var(--good); }
.c-running, .c-pending, .c-queued { background: var(--accent-soft); color: var(--accent); }
.c-uploading, .c-not_queued { background: var(--neutral-soft); color: var(--text-2); }
.c-missing_files { background: var(--warning-soft); color: var(--warning); }
.c-tiff_problem { background: var(--serious-soft); color: var(--serious); }
.c-failed { background: var(--critical-soft); color: var(--critical); }

/* Filters */
.filters { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; align-items: center; }
select, input[type=search] { font: inherit; font-size: 13px; padding: 6px 10px; border-radius: 7px; border: 1px solid var(--line-2); background: var(--surface); color: var(--text); }
input[type=search] { min-width: 220px; }
.check { font-size: 13px; color: var(--text-2); display: inline-flex; align-items: center; gap: 6px; margin-left: 4px; }
#msg { min-height: 20px; margin: 4px 0 8px; font-size: 13px; color: var(--text-2); }
#msg.err { color: var(--critical); }

/* Table */
.tablewrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); }
table { border-collapse: collapse; width: 100%; min-width: 1000px; }
th { text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: var(--text-3); font-weight: 600; background: var(--panel); padding: 9px 14px; position: sticky; top: 0; border-bottom: 1px solid var(--line); }
td { padding: 12px 14px; border-top: 1px solid var(--line); vertical-align: top; font-size: 13px; }
tbody tr:first-child td { border-top: 0; }
tbody tr:hover td { background: color-mix(in srgb, var(--panel) 55%, transparent); }
td.num, .num { font-variant-numeric: tabular-nums; }
.sid { font-weight: 650; font-size: 13.5px; white-space: nowrap; }
.sid .date { font-weight: 500; color: var(--text-2); margin-left: 6px; }
.muted { color: var(--text-3); }
.small { font-size: 12px; color: var(--text-2); }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }

/* File chips */
.chips { display: flex; flex-wrap: wrap; gap: 4px; }
.chip { display: inline-flex; align-items: center; gap: 4px; padding: 1px 8px; border-radius: 999px; font-size: 12px; white-space: nowrap; background: var(--neutral-soft); color: var(--text-2); }
.chip b { font-weight: 700; }
.chip.ok b { color: var(--good); } .chip.no { background: var(--critical-soft); color: var(--critical); }
.chip .dim { color: var(--text-3); }
.extra { margin-top: 5px; font-size: 11.5px; color: var(--text-3); }

/* Status */
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px 2px 8px; border-radius: 999px; font-weight: 600; font-size: 12.5px; white-space: nowrap; }
.pill .i { font-size: 12px; width: 1em; text-align: center; }
.pill.c-done, .pill.c-uploading, .pill.c-not_queued, .pill.c-running, .pill.c-pending, .pill.c-queued, .pill.c-missing_files, .pill.c-tiff_problem, .pill.c-failed { color: var(--text); }
.pill.c-done .i { color: var(--good); } .pill.c-running .i, .pill.c-pending .i, .pill.c-queued .i { color: var(--accent); }
.pill.c-uploading .i, .pill.c-not_queued .i { color: var(--text-3); } .pill.c-missing_files .i { color: var(--warning); }
.pill.c-tiff_problem .i { color: var(--serious); } .pill.c-failed .i { color: var(--critical); }
.detail.note { color: var(--warning); cursor: help; }
.detail { margin-top: 5px; font-size: 12px; color: var(--text-2); max-width: 46ch; }
a.detail { display: block; color: var(--text-2); text-decoration: underline; text-decoration-style: dotted; text-decoration-color: var(--text-3); text-underline-offset: 2px; }
a.detail:hover { color: var(--text); text-decoration-color: var(--accent); }
a.detail .go { color: var(--accent); text-decoration: none; display: inline-block; margin-left: 3px; }

/* QC */
.counts { font-size: 12.5px; white-space: nowrap; } .counts b { font-weight: 650; }
.flags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 5px; }
.flag { font-size: 11.5px; padding: 1px 8px; border-radius: 999px; background: var(--warning-soft); color: var(--text); border: 1px solid color-mix(in srgb, var(--warning) 45%, transparent); cursor: help; white-space: nowrap; }
.flag::before { content: "\26A0\FE0E  "; color: var(--warning); }
.qcok { color: var(--good); font-size: 12px; font-weight: 600; margin-top: 4px; }
.links { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 6px; font-size: 12.5px; font-weight: 500; }

/* Job */
.eta { color: var(--accent); white-space: nowrap; cursor: help; font-size: 12px; margin-top: 2px; }
.actions { display: flex; flex-direction: column; align-items: flex-start; gap: 6px; }
button.danger { border-color: color-mix(in srgb, var(--critical) 55%, transparent); color: var(--critical); background: var(--surface); }
button.danger:hover { background: var(--critical-soft); }
button.ghost { background: none; border-color: transparent; color: var(--text-3); font-size: 12px; padding: 0; font-weight: 500; }
button.ghost:hover { color: var(--text); text-decoration: underline; background: none; }
.empty { padding: 28px 14px; text-align: center; color: var(--text-3); }
@media (max-width: 640px) { main { padding: 18px 16px 40px; } h1 { font-size: 18px; } .mark { width: 32px; height: 32px; } input[type=search] { min-width: 0; flex: 1 1 160px; } .filters select { flex: 0 0 auto; } }
</style>
</head>
<body>
<main>
  <header>
    <div class="brand">
      <div class="mark" aria-hidden="true">2P</div>
      <div>
        <h1>Preprocessing pipeline</h1>
        <div class="sub" id="meta">Loading…</div>
      </div>
    </div>
    <div class="headright">
      <span class="small" id="lastscan"></span>
      <button class="primary" id="scan" title="Check all sessions now instead of waiting for the next scheduled scan">Scan now</button>
    </div>
  </header>
  <div class="tabs" role="tablist">
    <button class="tab" role="tab" data-tab="active" aria-selected="true">Sessions<span class="count" id="n-active"></span></button>
    <button class="tab" role="tab" data-tab="archive" aria-selected="false" title="All processed sessions; the Sessions tab only shows done sessions processed in the last 7 days">Archive<span class="count" id="n-archive"></span></button>
    <button class="tab minor" role="tab" data-tab="ignored" aria-selected="false" title="Sessions you chose to ignore; they are not processed automatically">Ignored<span class="count" id="n-ignored"></span></button>
  </div>
  <div class="tiles" id="tiles"></div>
  <div class="filters">
    <select id="mouse" aria-label="Mouse"><option value="">All mice</option></select>
    <input type="search" id="search" placeholder="Search date or session" aria-label="Search">
    <label class="check"><input type="checkbox" id="flagged"> Only sessions with QC flags</label>
  </div>
  <div id="msg" role="status"></div>
  <div class="tablewrap">
    <table>
      <thead><tr><th>Session</th><th>Files</th><th>Status</th><th>QC</th><th>Job</th><th></th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </div>
</main>
<script>
const STATUS = {
  failed:        {label: 'Failed',          icon: '✕'},
  tiff_problem:  {label: 'TIFF problem',    icon: '⚠︎'},
  running:       {label: 'Running',         icon: '▶︎'},
  pending:       {label: 'Pending',         icon: '◷'},
  queued:        {label: 'Queued',          icon: '•'},
  missing_files: {label: 'Missing files',   icon: '!'},
  uploading:     {label: 'Uploading',       icon: '↑'},
  not_queued:    {label: 'Not queued',      icon: '○'},
  done:          {label: 'Done',            icon: '✓'},
};
const ORDER = Object.keys(STATUS);
const PROBLEM = new Set(['missing_files', 'tiff_problem', 'uploading', 'failed']);  // their detail links to the file listing
let data = null, statusFilter = '', tab = 'active';
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const dateStr = d => `20${d.slice(0,2)}-${d.slice(2,4)}-${d.slice(4,6)}`;

function action(s) {
  if (s.status === 'not_queued') return ['Process', false, null];
  if (s.status === 'failed') return ['Retry', false, null];
  if (s.status === 'tiff_problem') return ['Process anyway', true, 'The last TIFF looks incomplete. Process anyway?'];
  if (s.status === 'missing_files' && s.n_tifs > 0 && s.sync && !s.virmen)
    return ['Process anyway', true, 'This session has no ViRMEn behavior file. Preprocess without behavior?'];
  if (s.status === 'done') return ['Reprocess', false, 'Reprocess this session? Its suite2p output and adata.h5ad will be overwritten.'];
  return null;
}

const mins = m => m >= 90 ? `${(m / 60).toFixed(1)} h` : `${Math.max(1, Math.round(m))} min`;
function etaText(s) {
  const e = s.eta;
  if (!e) return '';
  const tip = 'Estimated from the TIFF size and how long finished jobs took per GB';
  if (e.remaining_min != null) {
    if (e.remaining_min < 1) return `<div class="small eta" title="${tip}">finishing soon (estimate ${mins(e.total_min)})</div>`;
    const t = new Date(e.finish * 1000).toLocaleTimeString([], {hour: 'numeric', minute: '2-digit'});
    return `<div class="small eta" title="${tip}">~${mins(e.remaining_min)} left · done ~${t}</div>`;
  }
  return `<div class="small eta" title="${tip}">takes ~${mins(e.total_min)}</div>`;
}

function row(s) {
  const st = STATUS[s.status] || {label: s.status, icon: '?'};
  const chip = (ok, label, extra, tip) => `<span class="chip ${ok ? 'ok' : 'no'}" title="${tip}"><b>${ok ? '✓' : '✕'}</b>${label}${extra ? ` <span class="dim">${extra}</span>` : ''}</span>`;
  const files = `<div class="chips">${chip(s.n_tifs, 'TIFF', s.n_tifs ? `${s.n_tifs} · ${s.tif_gb} GB${s.n_compressed ? (s.n_compressed === s.n_tifs ? ' · zst' : ` · ${s.n_compressed} zst`) : ''}` : '', s.n_compressed ? 'ScanImage TIFFs, losslessly compressed to .tif.zst after processing' : 'ScanImage TIFFs')}${
      chip(s.virmen, 'ViRMEn', '', 'ViRMEn sessionData.mat')}${chip(s.sync, 'Sync', '', 'Sync file')}</div>${
      Object.keys(s.filter_stacks || {}).length ? `<div class="extra" title="Extra stacks in filter* subfolders, motion-corrected by the filters step">+ ${Object.keys(s.filter_stacks).map(esc).join(', ')}</div>` : ''}`;
  let qc = '<span class="muted">—</span>';
  if (s.qc_report || s.n_cells != null || s.movie) {
    const counts = s.n_cells != null ? `<div class="counts"><b>${s.n_cells.toLocaleString()}</b> cells · <b>${s.n_red.toLocaleString()}</b> R+${s.behavior_min ? ` · ${s.behavior_min} min` : ''}</div>` : '';
    const flags = s.qc_warnings.length
      ? `<div class="flags">${s.qc_warnings.map(w => `<span class="flag" title="${esc(w.message)}">${esc(w.label)}</span>`).join('')}</div>`
      : (s.n_cells != null ? '<div class="qcok">✓ No QC warnings</div>' : '');
    const link = (s.qc_report || s.movie) ? `<div class="links">${s.qc_report ? `<a href="/qc/${s.id}" target="_blank" rel="noopener">QC report ↗</a>` : ''}${
      s.movie ? `<a href="/movie/${s.id}" target="_blank" rel="noopener" title="Sped-up motion-corrected recording">Movie ↗</a>` : ''}</div>` : '';
    qc = counts + flags + link;
  }
  const job = s.job ? `<div class="mono">${esc(s.job.id)}</div><div class="small">${esc((s.job.state || '').toLowerCase())}${s.job.elapsed ? ' · ' + esc(s.job.elapsed) : ''}</div>` : (s.eta ? '' : '<span class="muted">—</span>');
  const eta = etaText(s);
  const log = s.has_log ? `<div><a class="small" href="/log/${s.id}" target="_blank" rel="noopener">Log ↗</a></div>` : '';
  const act = s.ignored ? null : action(s);
  const btn = act ? `<button class="act" data-op="queue" data-id="${esc(s.id)}" data-force="${act[1]}" data-confirm="${esc(act[2] || '')}">${act[0]}</button>` : '';
  const busy = s.status === 'running' || s.status === 'pending';
  const rmdir = s.empty_folder && !busy ? `<button class="danger" data-op="rmdir" data-id="${esc(s.id)}" title="The session's TIFF folder has no files, only empty subfolders. Remove it and drop the session from the dashboard.">Delete empty folder</button>` : '';
  const ign = s.ignored
    ? `<button data-op="restore" data-id="${esc(s.id)}" title="Move back to the Sessions tab">Restore</button>${
       (s.n_tifs || s.sync) ? `<button class="danger" data-op="delete" data-id="${esc(s.id)}" title="Permanently delete this session's TIFFs and sync file">Delete TIFFs/Sync</button>` : ''}`
    : (busy ? '' : `<button class="ghost" data-op="ignore" data-id="${esc(s.id)}" title="Move to the Ignored tab; it won't be processed automatically">Ignore</button>`);
  return `<tr>
    <td><div class="sid">${esc(s.mouse)}<span class="date">${dateStr(s.date)}</span></div><div class="small">${esc(s.session)}</div></td>
    <td>${files}</td>
    <td><span class="pill c-${esc(s.status)}"><span class="i" aria-hidden="true">${st.icon}</span>${st.label}</span>${
      PROBLEM.has(s.status) ? `<a class="detail" href="/files/${s.id}" target="_blank" rel="noopener" title="Open the list of files in this session's folders">${esc(s.detail)}<span class="go">↗</span></a>` : `${s.tif_notes && s.tif_notes.length ? `<div class="detail note" title="${esc(s.tif_notes.join(' '))}">ⓘ Last TIFF cut off; complete volumes are used</div>` : ''}<div class="detail">${esc(s.detail)}${s.status === 'done' && s.done_at ? ` · ${new Date(s.done_at * 1000).toLocaleString([], {month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit'})}` : ''}</div>`}</td>
    <td>${qc}</td>
    <td>${job}${eta}${log}</td>
    <td><div class="actions">${btn}${rmdir}${ign}</div></td></tr>`;
}

function render() {
  if (!data) return;
  $('meta').textContent = `New sessions recorded on or after ${dateStr(data.auto_start_date)} are processed automatically · up to ${data.max_concurrent_jobs} jobs at a time`;
  showLastScan();
  // Sessions: everything not ignored, except done sessions processed more than a week ago; Archive: all done sessions
  const weekAgo = Date.now() / 1000 - 7 * 86400;
  const recent = s => s.status !== 'done' || (s.done_at || 0) >= weekAgo;
  const active = data.sessions.filter(s => !s.ignored && recent(s)), ignored = data.sessions.filter(s => s.ignored);
  const archive = data.sessions.filter(s => !s.ignored && s.status === 'done');
  $('n-active').textContent = active.length; $('n-ignored').textContent = ignored.length; $('n-archive').textContent = archive.length;
  document.querySelectorAll('.tab').forEach(t => t.setAttribute('aria-selected', t.dataset.tab === tab));
  const shown = tab === 'ignored' ? ignored : tab === 'archive' ? archive : active;
  const counts = Object.fromEntries(ORDER.map(k => [k, 0]));
  active.forEach(s => counts[s.status] = (counts[s.status] || 0) + 1);
  const tiles = [['', 'All sessions', active.length], ...ORDER.filter(k => counts[k]).map(k => [k, k === 'done' ? 'Done this week' : STATUS[k].label, counts[k]])];
  $('tiles').hidden = tab !== 'active';
  $('tiles').innerHTML = tiles.map(([k, l, n]) => `<button class="tile" data-status="${k}" aria-pressed="${k === statusFilter}"><span class="ic c-${k || 'all'}" aria-hidden="true">${k ? STATUS[k].icon : '≡'}</span><span class="n">${n}</span><span class="l">${l}</span></button>`).join('');
  const mice = [...new Set(data.sessions.map(s => s.mouse))].sort();
  const mouseSel = $('mouse'), cur = mouseSel.value;
  mouseSel.innerHTML = '<option value="">All mice</option>' + mice.map(m => `<option ${m === cur ? 'selected' : ''}>${esc(m)}</option>`).join('');
  const q = $('search').value.trim().toLowerCase(), flagged = $('flagged').checked;
  const rows = shown
    .filter(s => (tab !== 'active' || !statusFilter || s.status === statusFilter) && (!mouseSel.value || s.mouse === mouseSel.value))
    .filter(s => !q || s.id.toLowerCase().includes(q) || dateStr(s.date).includes(q))
    .filter(s => !flagged || s.qc_warnings.length)
    .sort(tab === 'archive' ? (a, b) => (b.done_at || 0) - (a.done_at || 0) || a.id.localeCompare(b.id)
                            : (a, b) => ORDER.indexOf(a.status) - ORDER.indexOf(b.status) || b.date.localeCompare(a.date) || a.id.localeCompare(b.id));
  const none = tab === 'ignored' && !ignored.length ? 'No ignored sessions. Use <b>Ignore</b> on a session to move it here.'
    : tab === 'archive' && !archive.length ? 'No processed sessions yet.' : 'No sessions match.';
  $('rows').innerHTML = rows.length ? rows.map(row).join('') : `<tr><td colspan="6" class="empty">${none}</td></tr>`;
}

async function load() {
  try {
    const r = await fetch('/api/status'); if (!r.ok) throw new Error(await r.text());
    data = await r.json(); render();
  } catch (e) { message('Could not load status: ' + e.message, true); }
}
async function deleteRaw(b, id) {
  try {
    const r = await fetch('/api/delete_preview?id=' + encodeURIComponent(id)); const p = await r.json(); if (!r.ok) throw new Error(p.error);
    const what = [`${p.n_tifs} TIFF files (${p.tif_gb} GB) in:\n  ${p.folders.join('\n  ') || '(none)'}`,
                  `Sync file: ${p.sync.join(', ') || '(none)'}`].join('\n\n');
    const typed = prompt(`Permanently delete these files of ${id}?\n\n${what}\n\nViRMEn files and preprocessing output are kept. This can't be undone.\n\nType ${id} to confirm:`);
    if (typed === null) return;
    if (typed.trim() !== id) { message(`Not deleted: the name typed doesn't match ${id}.`, true); return; }
    b.disabled = true; message(`Deleting files of ${id}…`);
    const d = await fetch('/api/delete', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({id, confirm: typed.trim()})});
    const res = await d.json(); if (!d.ok) throw new Error(res.error);
    message(`${id}: deleted ${res.n_tifs} TIFFs and ${res.n_sync} sync file(s), ${res.gb} GB.`);
  } catch (e) { message(`${id}: ${e.message}`, true); }
  load();
}
function ago(t) {
  const m = Math.round((Date.now() / 1000 - t) / 60);
  return m < 1 ? 'just now' : m < 60 ? `${m} min ago` : m < 1440 ? `${Math.round(m / 60)} h ago` : new Date(t * 1000).toLocaleString();
}
function showLastScan() {
  if (!data) return;
  $('lastscan').innerHTML = data.last_scan ? `<span class="live" aria-hidden="true"></span>Last scan ${ago(data.last_scan)}` : 'Not scanned yet';
  $('lastscan').title = data.last_scan ? `${new Date(data.last_scan * 1000).toLocaleString()}; scans run every 10 minutes` : '';
}
setInterval(showLastScan, 30000);
function message(text, err) { const m = $('msg'); m.textContent = text; m.className = err ? 'err' : ''; }

document.addEventListener('click', async ev => {
  const tile = ev.target.closest('.tile');
  if (tile) { statusFilter = tile.dataset.status === statusFilter ? '' : tile.dataset.status; render(); return; }
  const t = ev.target.closest('.tab');
  if (t) { tab = t.dataset.tab; render(); return; }
  const b = ev.target.closest('button[data-id]');
  if (!b) return;
  if (b.dataset.confirm && !confirm(b.dataset.confirm)) return;
  const op = b.dataset.op, id = b.dataset.id;
  if (op === 'delete') return deleteRaw(b, id);
  if (op === 'rmdir') {
    if (!confirm(`Remove the empty folder of ${id}? It contains no files, only empty subfolders. The session will disappear from the dashboard.`)) return;
    b.disabled = true;
    try {
      const r = await fetch('/api/delete_folder', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({id})});
      const res = await r.json(); if (!r.ok) throw new Error(res.error);
      message(`Removed ${res.folder}.`);
    } catch (e) { message(`${id}: ${e.message}`, true); }
    return load();
  }
  b.disabled = true; message(op === 'queue' ? `Submitting ${id}…` : op === 'ignore' ? `Ignoring ${id}…` : `Restoring ${id}…`);
  try {
    const [url, body] = op === 'queue' ? ['/api/queue', {id, force: b.dataset.force === 'true'}] : ['/api/ignore', {id, ignored: op === 'ignore'}];
    const r = await fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    const res = await r.json(); if (!r.ok) throw new Error(res.error);
    message(op === 'queue' ? `${id}: ${STATUS[res.status]?.label || res.status}. ${res.detail}`
      : op === 'ignore' ? `${id} moved to Ignored.` : `${id} restored: ${STATUS[res.status]?.label || res.status}.`);
  } catch (e) { message(`${id}: ${e.message}`, true); }
  load();
});
$('scan').addEventListener('click', async () => {
  const b = $('scan'); b.disabled = true; message('Scanning sessions…');
  try { const r = await fetch('/api/scan', {method: 'POST'}); data = await r.json(); render(); message('Scan complete.'); }
  catch (e) { message('Scan failed: ' + e.message, true); }
  b.disabled = false;
});
['mouse', 'search', 'flagged'].forEach(id => $(id).addEventListener('input', render));
load(); setInterval(load, 30000);
</script>
</body>
</html>
"""

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Preprocessing pipeline dashboard.')
    parser.add_argument('--port', type=int, default=8050)
    parser.add_argument('--url', help='public address to print, e.g. the dev tunnel URL')
    args = parser.parse_args()
    main(args.port, args.url)
