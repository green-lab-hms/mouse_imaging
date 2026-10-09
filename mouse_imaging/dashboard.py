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
import json, re, secrets, mimetypes
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
    sessions = []
    for sid, e in sorted(state['sessions'].items()):
        files, out = e.get('files', {}), e.get('outputs', {})
        qc = out.get('qc') or {}
        sessions.append({
            'id': sid, 'mouse': e['mouse'], 'date': e['date'], 'session': e['session'],
            'status': e.get('status'), 'detail': e.get('detail'), 'ignored': bool(e.get('ignored')),
            'n_tifs': files.get('n_tifs', 0), 'tif_gb': round(files.get('tif_bytes', 0) / 1e9, 1),
            'virmen': files.get('virmen', False), 'sync': bool(files.get('sync')),
            'job': e.get('job'), 'has_log': bool(out.get('log')), 'qc_report': out.get('qc_report', False),
            'qc_warnings': qc.get('warnings', []), 'n_cells': qc.get('n_cells'), 'n_red': qc.get('n_red'),
            'behavior_min': qc.get('behavior_min'),
        })
    s = pipeline.settings()
    return {'last_scan': state.get('last_scan'), 'auto_start_date': state.get('auto_start_date'),
            'max_concurrent_jobs': s['max_concurrent_jobs'], 'quiet_minutes': s['quiet_minutes'], 'sessions': sessions}

def session_parts(path_parts):
    if len(path_parts) != 3 or not all(PART_RE.match(p) for p in path_parts) or not pipeline.SESSION_RE.match(path_parts[2]):
        raise ValueError('bad session path')
    return path_parts

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
        except ValueError:
            return self._send(400, 'Bad request', 'text/plain')
        self._send(404, 'Not found', 'text/plain')

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
  --surface: #fcfcfb; --panel: #f4f3f0; --line: #e6e5e1;
  --text: #0b0b0b; --text-2: #52514e; --text-3: #8a8985;
  --accent: #2a78d6; --accent-ink: #ffffff;
  --good: #0ca30c; --warning: #b77a00; --warning-bg: #fdf3dc; --serious: #c4561f; --critical: #d03b3b; --critical-bg: #fbe9e9;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface: #1a1a19; --panel: #242422; --line: #383835;
    --text: #ffffff; --text-2: #c3c2b7; --text-3: #8f8e86;
    --accent: #3987e5; --accent-ink: #ffffff;
    --good: #2fbf2f; --warning: #fab219; --warning-bg: #3a3017; --serious: #ec835a; --critical: #e66767; --critical-bg: #3b2121;
  }
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
body { margin: 0; background: var(--surface); color: var(--text); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 24px 16px 48px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 16px; justify-content: space-between; }
h1 { font-size: 22px; margin: 0; }
.sub { color: var(--text-2); font-size: 13px; }
button { font: inherit; font-size: 13px; border-radius: 6px; border: 1px solid var(--line); background: var(--panel); color: var(--text); padding: 5px 12px; cursor: pointer; white-space: nowrap; }
button.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
button:disabled { opacity: .5; cursor: progress; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin: 20px 0 16px; }
.tile { background: var(--panel); border-radius: 8px; padding: 10px 12px; border: 2px solid transparent; cursor: pointer; text-align: left; }
.tile[aria-pressed="true"] { border-color: var(--accent); }
.tile .n { font-size: 22px; font-weight: 700; display: block; }
.tile .l { font-size: 12px; color: var(--text-2); }
.filters { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 12px; align-items: center; }
select, input[type=search] { font: inherit; font-size: 13px; padding: 5px 8px; border-radius: 6px; border: 1px solid var(--line); background: var(--surface); color: var(--text); }
.tablewrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; min-width: 980px; }
th { text-align: left; font-size: 12px; color: var(--text-2); font-weight: 600; background: var(--panel); padding: 8px 10px; position: sticky; top: 0; }
td { padding: 9px 10px; border-top: 1px solid var(--line); vertical-align: top; font-size: 13px; }
td.num { font-variant-numeric: tabular-nums; }
.sid { font-weight: 600; }
td:first-child { white-space: nowrap; }
td:first-child .small { white-space: nowrap; }
.muted { color: var(--text-3); }
.small { font-size: 12px; color: var(--text-2); }
.files span { display: inline-block; margin-right: 10px; white-space: nowrap; }
.ok { color: var(--good); font-weight: 700; } .no { color: var(--critical); font-weight: 700; }
.badge { display: inline-flex; align-items: center; gap: 5px; font-weight: 600; white-space: nowrap; }
.badge .i { font-size: 13px; width: 1.1em; text-align: center; }
.s-done .i { color: var(--good); } .s-running .i, .s-pending .i, .s-queued .i { color: var(--accent); }
.s-uploading .i, .s-not_queued .i { color: var(--text-3); } .s-missing_files .i { color: var(--warning); }
.s-tiff_problem .i { color: var(--serious); } .s-failed .i { color: var(--critical); }
.flags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.flag { font-size: 11.5px; padding: 1px 7px; border-radius: 999px; background: var(--warning-bg); color: var(--text); border: 1px solid var(--warning); cursor: help; white-space: nowrap; }
.flag::before { content: "\26A0\FE0E  "; color: var(--warning); }
.qcok { color: var(--good); font-size: 12px; font-weight: 600; }
a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
.tabs { display: flex; align-items: flex-end; gap: 4px; border-bottom: 1px solid var(--line); margin-bottom: 12px; }
.tab { background: none; border: 0; border-bottom: 2px solid transparent; border-radius: 0; padding: 6px 10px; color: var(--text-2); font-weight: 600; margin-bottom: -1px; }
.tab[aria-selected="true"] { color: var(--text); border-bottom-color: var(--accent); }
.tab.minor { margin-left: auto; font-size: 12px; font-weight: 500; }
.tab .count { color: var(--text-3); font-weight: 500; margin-left: 4px; }
.actions { display: flex; flex-direction: column; align-items: flex-start; gap: 4px; }
button.ghost { background: none; border-color: transparent; color: var(--text-2); font-size: 12px; padding: 2px 6px; margin-left: -6px; }
button.ghost:hover { color: var(--text); text-decoration: underline; }
#msg { min-height: 20px; margin: 8px 0; font-size: 13px; }
#msg.err { color: var(--critical); }
@media (max-width: 640px) { main { padding: 16px; } h1 { font-size: 19px; } }
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>Preprocessing pipeline</h1>
      <div class="sub" id="meta">Loading…</div>
    </div>
    <button class="primary" id="scan" title="Check all sessions now instead of waiting for the next scheduled scan">Scan now</button>
  </header>
  <div class="tabs" role="tablist">
    <button class="tab" role="tab" data-tab="active" aria-selected="true">Sessions<span class="count" id="n-active"></span></button>
    <button class="tab minor" role="tab" data-tab="ignored" aria-selected="false" title="Sessions you chose to ignore; they are not processed automatically">Ignored<span class="count" id="n-ignored"></span></button>
  </div>
  <div class="tiles" id="tiles"></div>
  <div class="filters">
    <select id="mouse" aria-label="Mouse"><option value="">All mice</option></select>
    <input type="search" id="search" placeholder="Search date or session" aria-label="Search">
    <label class="small"><input type="checkbox" id="flagged"> Only sessions with QC flags</label>
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

function row(s) {
  const st = STATUS[s.status] || {label: s.status, icon: '?'};
  const files = `<div class="files">
      <span title="ScanImage TIFFs">${s.n_tifs ? '<span class="ok">✓</span>' : '<span class="no">✕</span>'} TIFF ${s.n_tifs ? `<span class="muted">${s.n_tifs} · ${s.tif_gb} GB</span>` : ''}</span>
      <span title="ViRMEn sessionData.mat">${s.virmen ? '<span class="ok">✓</span>' : '<span class="no">✕</span>'} ViRMEn</span>
      <span title="Sync file">${s.sync ? '<span class="ok">✓</span>' : '<span class="no">✕</span>'} Sync</span></div>`;
  let qc = '<span class="muted">—</span>';
  if (s.qc_report || s.n_cells != null) {
    const counts = s.n_cells != null ? `<div>${s.n_cells.toLocaleString()} cells · ${s.n_red.toLocaleString()} R+${s.behavior_min ? ` · ${s.behavior_min} min` : ''}</div>` : '';
    const flags = s.qc_warnings.length
      ? `<div class="flags">${s.qc_warnings.map(w => `<span class="flag" title="${esc(w.message)}">${esc(w.label)}</span>`).join('')}</div>`
      : (s.n_cells != null ? '<div class="qcok">✓ No QC warnings</div>' : '');
    const link = s.qc_report ? `<div><a href="/qc/${s.id}" target="_blank" rel="noopener">QC report ↗</a></div>` : '';
    qc = counts + flags + link;
  }
  const job = s.job ? `<div class="num">${esc(s.job.id)}</div><div class="small">${esc((s.job.state || '').toLowerCase())}${s.job.elapsed ? ' · ' + esc(s.job.elapsed) : ''}</div>` : '<span class="muted">—</span>';
  const log = s.has_log ? `<div><a class="small" href="/log/${s.id}" target="_blank" rel="noopener">Log ↗</a></div>` : '';
  const act = s.ignored ? null : action(s);
  const btn = act ? `<button data-op="queue" data-id="${esc(s.id)}" data-force="${act[1]}" data-confirm="${esc(act[2] || '')}">${act[0]}</button>` : '';
  const busy = s.status === 'running' || s.status === 'pending';
  const ign = s.ignored
    ? `<button data-op="restore" data-id="${esc(s.id)}" title="Move back to the Sessions tab">Restore</button>`
    : (busy ? '' : `<button class="ghost" data-op="ignore" data-id="${esc(s.id)}" title="Move to the Ignored tab; it won't be processed automatically">Ignore</button>`);
  return `<tr>
    <td><div class="sid">${esc(s.mouse)} <span class="muted">·</span> ${dateStr(s.date)}</div><div class="small">${esc(s.session)}</div></td>
    <td>${files}</td>
    <td><span class="badge s-${esc(s.status)}"><span class="i" aria-hidden="true">${st.icon}</span>${st.label}</span><div class="small">${esc(s.detail)}</div></td>
    <td>${qc}</td>
    <td>${job}${log}</td>
    <td><div class="actions">${btn}${ign}</div></td></tr>`;
}

function render() {
  if (!data) return;
  const last = data.last_scan ? new Date(data.last_scan * 1000).toLocaleString() : 'never';
  $('meta').textContent = `Last scan ${last} · New sessions recorded on or after ${dateStr(data.auto_start_date)} are processed automatically · Up to ${data.max_concurrent_jobs} jobs at a time`;
  const active = data.sessions.filter(s => !s.ignored), ignored = data.sessions.filter(s => s.ignored);
  $('n-active').textContent = active.length; $('n-ignored').textContent = ignored.length;
  document.querySelectorAll('.tab').forEach(t => t.setAttribute('aria-selected', t.dataset.tab === tab));
  const shown = tab === 'ignored' ? ignored : active;
  const counts = Object.fromEntries(ORDER.map(k => [k, 0]));
  active.forEach(s => counts[s.status] = (counts[s.status] || 0) + 1);
  const tiles = [['', 'All sessions', active.length], ...ORDER.filter(k => counts[k]).map(k => [k, STATUS[k].label, counts[k]])];
  $('tiles').hidden = tab === 'ignored';
  $('tiles').innerHTML = tiles.map(([k, l, n]) => `<button class="tile" data-status="${k}" aria-pressed="${k === statusFilter}"><span class="n">${n}</span><span class="l">${l}</span></button>`).join('');
  const mice = [...new Set(data.sessions.map(s => s.mouse))].sort();
  const mouseSel = $('mouse'), cur = mouseSel.value;
  mouseSel.innerHTML = '<option value="">All mice</option>' + mice.map(m => `<option ${m === cur ? 'selected' : ''}>${esc(m)}</option>`).join('');
  const q = $('search').value.trim().toLowerCase(), flagged = $('flagged').checked;
  const rows = shown
    .filter(s => (tab === 'ignored' || !statusFilter || s.status === statusFilter) && (!mouseSel.value || s.mouse === mouseSel.value))
    .filter(s => !q || s.id.toLowerCase().includes(q) || dateStr(s.date).includes(q))
    .filter(s => !flagged || s.qc_warnings.length)
    .sort((a, b) => ORDER.indexOf(a.status) - ORDER.indexOf(b.status) || b.date.localeCompare(a.date) || a.id.localeCompare(b.id));
  const none = tab === 'ignored' && !ignored.length ? 'No ignored sessions. Use <b>Ignore</b> on a session to move it here.' : 'No sessions match.';
  $('rows').innerHTML = rows.length ? rows.map(row).join('') : `<tr><td colspan="6" class="muted">${none}</td></tr>`;
}

async function load() {
  try {
    const r = await fetch('/api/status'); if (!r.ok) throw new Error(await r.text());
    data = await r.json(); render();
  } catch (e) { message('Could not load status: ' + e.message, true); }
}
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
