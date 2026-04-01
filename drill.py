import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from drill_core import evaluate_exercise, generate_exercise, scan_drills

PORT = 7234

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>drill</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:         #0f1117;
  --bg-side:    #161b27;
  --bg-alt:     #111520;
  --bg-blank:   rgba(88,166,255,0.06);
  --bg-correct: rgba(63,185,80,0.08);
  --bg-wrong:   rgba(248,81,73,0.08);
  --border:     #21262d;
  --text:       #c9d1d9;
  --text-dim:   #484f58;
  --text-muted: #6e7681;
  --accent:     #58a6ff;
  --green:      #3fb950;
  --red:        #f85149;
  --radius:     6px;
  --font-code:  'JetBrains Mono','Fira Code',monospace;
  --font-ui:    'Inter',system-ui,sans-serif;
}
html,body { height:100%; background:var(--bg); color:var(--text); font-family:var(--font-ui); }

.app { display:flex; height:100vh; overflow:hidden; }

/* sidebar */
.sidebar {
  width:260px; min-width:260px;
  background:var(--bg-side); border-right:1px solid var(--border);
  display:flex; flex-direction:column; overflow-y:auto;
}
.sidebar-hd { padding:24px 20px 18px; border-bottom:1px solid var(--border); }
.sidebar-hd h1 { font-family:var(--font-code); font-size:20px; font-weight:600; color:var(--accent); letter-spacing:-0.5px; }
.sidebar-hd p  { font-size:11px; color:var(--text-muted); margin-top:3px; }

.controls { padding:20px; display:flex; flex-direction:column; gap:16px; flex:1; }
.field { display:flex; flex-direction:column; gap:6px; }
.field label { font-size:11px; font-weight:600; color:var(--text-muted); text-transform:uppercase; letter-spacing:.6px; }
.field select {
  background:var(--bg); border:1px solid var(--border); color:var(--text);
  font-family:var(--font-ui); font-size:13px; padding:7px 10px;
  border-radius:var(--radius); width:100%; outline:none; cursor:pointer;
  transition:border-color .15s;
}
.field select:focus { border-color:var(--accent); }

.mask-row { display:flex; align-items:center; gap:10px; }
.mask-row input[type=range] { flex:1; accent-color:var(--accent); cursor:pointer; }
.mask-val { font-family:var(--font-code); font-size:12px; color:var(--accent); min-width:34px; text-align:right; }

.toggle-row { display:flex; align-items:center; justify-content:space-between; }
.toggle-row > span { font-size:13px; color:var(--text); }
.toggle { position:relative; width:36px; height:20px; cursor:pointer; flex-shrink:0; }
.toggle input { opacity:0; width:0; height:0; }
.t-track { position:absolute; inset:0; background:var(--border); border-radius:20px; transition:background .2s; }
.toggle input:checked + .t-track { background:var(--accent); }
.t-thumb { position:absolute; top:3px; left:3px; width:14px; height:14px; border-radius:50%; background:#fff; transition:transform .2s; pointer-events:none; }
.toggle input:checked ~ .t-thumb { transform:translateX(16px); }

.btn-gen {
  background:var(--accent); color:#0f1117;
  border:none; border-radius:var(--radius);
  font-family:var(--font-ui); font-size:13px; font-weight:600;
  padding:9px 14px; cursor:pointer; width:100%;
  transition:opacity .15s,transform .1s;
}
.btn-gen:hover { opacity:.88; }
.btn-gen:active { transform:scale(.98); }
.btn-gen:disabled { opacity:.5; cursor:default; }

.score-box {
  margin:0 20px 20px; padding:14px 16px;
  background:var(--bg); border:1px solid var(--border); border-radius:var(--radius);
  display:none;
}
.score-box.on { display:block; }
.score-num { font-family:var(--font-code); font-size:26px; font-weight:600; color:var(--text); }
.score-num.perfect { color:var(--green); }
.score-sub { font-size:11px; color:var(--text-muted); margin-top:2px; }

/* editor */
.editor { flex:1; display:flex; flex-direction:column; overflow:hidden; }

.src-bar {
  display:none; align-items:center; justify-content:space-between;
  padding:0 20px; height:38px;
  background:var(--bg-side); border-bottom:1px solid var(--border);
  font-size:12px;
}
.src-bar.on { display:flex; }
.src-label { font-family:var(--font-code); color:var(--text-muted); }
.src-label b { color:var(--accent); font-weight:500; }
.blank-pill {
  font-family:var(--font-code); font-size:11px; color:var(--text-dim);
  background:var(--bg); border:1px solid var(--border); padding:2px 8px; border-radius:10px;
}

.empty {
  flex:1; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px;
  color:var(--text-muted);
}
.empty .icon { font-family:var(--font-code); font-size:36px; color:var(--border); line-height:1; }
.empty p { font-size:13px; }
.empty kbd {
  background:var(--bg-side); border:1px solid var(--border);
  padding:1px 6px; border-radius:4px; font-family:var(--font-code); font-size:11px;
}

.code-scroll { flex:1; overflow-y:auto; overflow-x:auto; padding:12px 0 100px; }

/* code lines */
.cl {
  display:flex; align-items:stretch; min-height:22px;
  font-family:var(--font-code); font-size:13px; line-height:22px;
}
.cl:nth-child(even) { background:var(--bg-alt); }
.ln {
  width:52px; min-width:52px; padding-right:16px; text-align:right;
  color:var(--text-dim); font-size:12px; line-height:22px; user-select:none;
}
.lb { flex:1; white-space:pre; padding-right:20px; }

.cl.bl { background:var(--bg-blank) !important; border-left:2px solid var(--accent); }
.cl.bl .ln { color:var(--accent); }
.b-indent { display:inline; white-space:pre; color:transparent; user-select:none; }
.b-in {
  background:transparent; border:none; outline:none;
  font-family:var(--font-code); font-size:13px; line-height:22px;
  color:#9cdcfe; caret-color:var(--accent);
  flex:1; min-width:200px;
}
.b-in::placeholder { color:var(--text-dim); font-style:italic; }

.cl.ok { background:var(--bg-correct) !important; border-left:2px solid var(--green); }
.cl.ok .ln { color:var(--green); }
.cl.ok .b-in { color:var(--green); }

.cl.no { background:var(--bg-wrong) !important; border-left:2px solid var(--red); }
.cl.no .ln { color:var(--red); }
.cl.no .b-in { color:var(--red); text-decoration:line-through; }

.ans-row {
  display:flex; align-items:stretch; min-height:21px;
  background:rgba(63,185,80,0.05); border-left:2px solid var(--green);
  font-family:var(--font-code); font-size:12px; line-height:21px;
}
.ans-row .ln { color:var(--green); font-size:11px; }
.ans-txt { color:var(--green); white-space:pre; padding-right:20px; }

/* eval bar */
.eval-bar {
  display:none; position:sticky; bottom:0;
  padding:14px 20px;
  background:linear-gradient(to top,var(--bg) 60%,transparent);
  justify-content:flex-end; align-items:center; gap:14px;
}
.eval-bar.on { display:flex; }
.eval-hint { font-size:11px; color:var(--text-muted); }
.eval-hint kbd {
  background:var(--bg-side); border:1px solid var(--border);
  padding:1px 5px; border-radius:3px; font-family:var(--font-code); font-size:10px;
}
.btn-eval {
  background:#238636; color:#fff; border:1px solid #2ea043;
  border-radius:var(--radius); font-family:var(--font-ui); font-size:13px; font-weight:600;
  padding:8px 22px; cursor:pointer; transition:background .15s,transform .1s;
}
.btn-eval:hover { background:#2ea043; }
.btn-eval:active { transform:scale(.98); }
.btn-eval:disabled { opacity:.5; cursor:default; }

::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="sidebar-hd">
      <h1>drill</h1>
      <p>fill in the blanks</p>
    </div>
    <div class="controls">
      <div class="field">
        <label>Exercise</label>
        <select id="target-sel"><option value="">Random</option></select>
      </div>
      <div class="field">
        <label>Mask</label>
        <div class="mask-row">
          <input type="range" id="mask" min="0.05" max="1" step="0.05" value="0.4">
          <span class="mask-val" id="mask-val">40%</span>
        </div>
      </div>
      <div class="field">
        <label>Guidance hints</label>
        <div class="toggle-row">
          <span>Show <code style="font-size:11px;color:var(--text-muted)"># code here</code></span>
          <label class="toggle">
            <input type="checkbox" id="guidance" checked>
            <div class="t-track"></div>
            <div class="t-thumb"></div>
          </label>
        </div>
      </div>
      <button class="btn-gen" id="gen-btn">Generate Exercise →</button>
    </div>
    <div class="score-box" id="score-box">
      <div class="score-num" id="score-num">—</div>
      <div class="score-sub" id="score-sub">lines correct</div>
    </div>
  </aside>

  <main class="editor">
    <div class="src-bar" id="src-bar">
      <span class="src-label" id="src-label"></span>
      <span class="blank-pill" id="blank-pill"></span>
    </div>
    <div class="empty" id="empty">
      <div class="icon">{ }</div>
      <p>Configure an exercise and click <kbd>Generate</kbd></p>
    </div>
    <div class="code-scroll" id="code-scroll" style="display:none">
      <div id="code-view"></div>
    </div>
    <div class="eval-bar" id="eval-bar">
      <span class="eval-hint">or press <kbd>Ctrl+Enter</kbd></span>
      <button class="btn-eval" id="eval-btn">Evaluate →</button>
    </div>
  </main>
</div>

<script>
let ex = null;

async function loadItems() {
  const r = await fetch('/api/items');
  const items = await r.json();
  const sel = document.getElementById('target-sel');
  const byFile = {};
  items.forEach(it => {
    const f = it.file.replace(/^drills\//, '');
    (byFile[f] = byFile[f] || []).push(it);
  });
  for (const [file, its] of Object.entries(byFile)) {
    const grp = document.createElement('optgroup');
    grp.label = file;
    its.forEach(it => {
      const o = document.createElement('option');
      o.value = it.label;
      o.textContent = it.name + (it.type === 'class' ? ' (class)' : '');
      grp.appendChild(o);
    });
    sel.appendChild(grp);
  }
}

document.getElementById('mask').addEventListener('input', function() {
  document.getElementById('mask-val').textContent = Math.round(+this.value * 100) + '%';
});

document.getElementById('gen-btn').addEventListener('click', generate);
async function generate() {
  const btn = document.getElementById('gen-btn');
  btn.textContent = 'Generating…'; btn.disabled = true;
  try {
    const r = await fetch('/api/drill', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        target: document.getElementById('target-sel').value || null,
        mask:   +document.getElementById('mask').value,
        guidance: document.getElementById('guidance').checked
      })
    });
    const data = await r.json();
    if (data.error) { alert(data.error); return; }
    ex = data;
    render(ex);
    document.getElementById('score-box').classList.remove('on');
  } catch(e) { alert(e); }
  finally { btn.textContent = 'Generate Exercise →'; btn.disabled = false; }
}

function render(ex) {
  document.getElementById('empty').style.display = 'none';
  document.getElementById('code-scroll').style.display = 'block';
  document.getElementById('eval-bar').classList.add('on');
  document.getElementById('src-bar').classList.add('on');

  const parts = ex.source.split('::');
  document.getElementById('src-label').innerHTML =
    `<span style="opacity:.5">${parts[0]}::</span><b>${parts[1]}</b>`;
  document.getElementById('blank-pill').textContent =
    ex.num_blanks + (ex.num_blanks === 1 ? ' blank' : ' blanks');

  const view = document.getElementById('code-view');
  view.innerHTML = '';

  ex.lines.forEach(line => {
    const row = document.createElement('div');

    if (line.blank) {
      row.className = 'cl bl';
      row.dataset.bid = line.bid;

      const ln = document.createElement('div');
      ln.className = 'ln'; ln.textContent = line.n;

      const lb = document.createElement('div');
      lb.className = 'lb'; lb.style.display = 'flex';

      const sp = document.createElement('span');
      sp.className = 'b-indent'; sp.textContent = line.indent || '';

      const inp = document.createElement('input');
      inp.className = 'b-in'; inp.type = 'text';
      inp.dataset.bid = line.bid;
      inp.placeholder = line.placeholder || '';
      inp.spellcheck = false; inp.autocomplete = 'off';
      inp.addEventListener('keydown', e => {
        if (e.key === 'Tab') {
          e.preventDefault();
          const all = [...document.querySelectorAll('.b-in')];
          const i = all.indexOf(e.target);
          const next = all[e.shiftKey ? i-1 : i+1];
          if (next) next.focus();
        }
      });

      lb.appendChild(sp); lb.appendChild(inp);
      row.appendChild(ln); row.appendChild(lb);
    } else {
      row.className = 'cl';
      const ln = document.createElement('div');
      ln.className = 'ln'; ln.textContent = line.n;
      const lb = document.createElement('div');
      lb.className = 'lb'; lb.textContent = line.text;
      row.appendChild(ln); row.appendChild(lb);
    }
    view.appendChild(row);
  });

  const first = view.querySelector('.b-in');
  if (first) setTimeout(() => first.focus(), 50);
}

document.getElementById('eval-btn').addEventListener('click', evaluate);
document.addEventListener('keydown', e => { if ((e.ctrlKey||e.metaKey) && e.key==='Enter') evaluate(); });

async function evaluate() {
  if (!ex) return;
  const answers = {};
  document.querySelectorAll('.b-in').forEach(i => { answers[i.dataset.bid] = i.value; });

  const btn = document.getElementById('eval-btn');
  btn.textContent = 'Checking…'; btn.disabled = true;

  try {
    const r = await fetch('/api/eval', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({id: ex.id, answers})
    });
    const res = await r.json();
    if (res.error) { alert(res.error); return; }
    showResults(res);
  } catch(e) { alert(e); }
  finally { btn.textContent = 'Evaluate →'; btn.disabled = false; }
}

function showResults(res) {
  const byBid = {};
  res.results.forEach(r => { byBid[r.bid] = r; });

  document.querySelectorAll('.ans-row').forEach(el => el.remove());

  document.querySelectorAll('.cl.bl').forEach(row => {
    const bid = +row.dataset.bid;
    const result = byBid[bid];
    if (!result) return;

    const inp = row.querySelector('.b-in');
    inp.disabled = true;

    if (result.correct) {
      row.classList.add('ok');
    } else {
      row.classList.add('no');
      const hint = document.createElement('div');
      hint.className = 'ans-row';

      const ln = document.createElement('div');
      ln.className = 'ln'; ln.textContent = '✓';

      const txt = document.createElement('div');
      txt.className = 'ans-txt';
      const line = ex.lines.find(l => l.blank && l.bid === bid);
      txt.textContent = (line ? line.indent : '') + result.answer;

      hint.appendChild(ln); hint.appendChild(txt);
      row.insertAdjacentElement('afterend', hint);
    }
  });

  const [got, total] = res.score.split('/').map(Number);
  const box = document.getElementById('score-box');
  const num = document.getElementById('score-num');
  const sub = document.getElementById('score-sub');
  num.textContent = res.score;
  num.className = 'score-num' + (res.all_correct ? ' perfect' : '');
  sub.textContent = res.all_correct ? 'perfect!' : `of ${total} correct`;
  box.classList.add('on');

  const first = document.querySelector('.cl.no');
  if (first) first.scrollIntoView({behavior:'smooth', block:'center'});
}

loadItems();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/items":
            try:
                self.send_json(scan_drills())
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
        else:
            self.send_response(404)
            self.end_headers()

    def read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n))

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self.read_body()
            if path == "/api/drill":
                result = generate_exercise(
                    body.get("target"),
                    float(body.get("mask", 0.4)),
                    bool(body.get("guidance", True)),
                )
                self.send_json(result)
            elif path == "/api/eval":
                result = evaluate_exercise(body["id"], body["answers"])
                self.send_json(result)
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            self.send_json({"error": str(e)}, 500)


def main():
    if not os.path.exists("drills"):
        print("Error: 'drills/' not found. Run from the transformer_drills root.")
        sys.exit(1)
    url = f"http://localhost:{PORT}"
    print(f"\n  drill UI  →  {url}")
    print(  "  Ctrl+C to stop\n")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    server = HTTPServer(("localhost", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")


if __name__ == "__main__":
    main()