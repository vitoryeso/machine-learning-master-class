"""
Accessible HTML annotator for hierarchical visual clusters.

Usage:
  python cluster_annotator.py
  python cluster_annotator.py --hierarchy output/hierarchical/hierarchy.json --port 8788

Open:
  http://localhost:8788

The app auto-saves cluster annotations to cluster_annotations.json after every
change and refreshes remote changes periodically, so multiple browser windows
can annotate with near real-time synchronization.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List
import urllib.parse


ANNOTATIONS_FILE_DEFAULT = "cluster_annotations.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster annotation web server.")
    p.add_argument("--hierarchy", default="output/hierarchical/hierarchy.json")
    p.add_argument("--paths", default="all_paths.txt")
    p.add_argument("--viz-paths", default="all_paths_256.txt")
    p.add_argument("--annotations", default=ANNOTATIONS_FILE_DEFAULT)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8788)
    return p.parse_args()


def load_json(path: str | Path, default: Any) -> Any:
    if Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: str | Path, obj: Any) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_lines(path: str | Path) -> List[str]:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def flatten_leaves(hierarchy: Dict[str, Any]) -> List[Dict[str, Any]]:
    leaves: List[Dict[str, Any]] = []
    for macro in hierarchy.get("clusters", []):
        for leaf in macro.get("subclusters", []):
            item = dict(leaf)
            item["macro_summary"] = macro.get("summary", {})
            leaves.append(item)
    leaves.sort(key=lambda x: x.get("id", ""))
    return leaves


def response_json(handler: SimpleHTTPRequestHandler, obj: Any, code: int = 200) -> None:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class ClusterAnnotatorHandler(SimpleHTTPRequestHandler):
    hierarchy_path = ""
    annotations_path = ""
    paths: List[str] = []
    viz_paths: List[str] = []
    leaves: List[Dict[str, Any]] = []

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[annotator]", fmt % args)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/state":
            annotations = load_json(self.annotations_path, {})
            response_json(
                self,
                {
                    "hierarchy_path": self.hierarchy_path,
                    "annotations_path": self.annotations_path,
                    "total": len(self.leaves),
                    "annotated": len(annotations),
                    "clusters": self.leaves,
                    "annotations": annotations,
                },
            )
            return

        if path.startswith("/api/image/"):
            try:
                idx = int(path.rsplit("/", 1)[-1])
                source = self.viz_paths[idx] if idx < len(self.viz_paths) and self.viz_paths else self.paths[idx]
                self.serve_file(source)
            except Exception as e:
                response_json(self, {"error": str(e)}, 404)
            return

        response_json(self, {"error": "not found"}, 404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/annotation":
            response_json(self, {"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            cluster_id = str(payload["cluster_id"])
            annotation = {
                "label_final": str(payload.get("label_final", "")).strip(),
                "usable_for_classification": bool(payload.get("usable_for_classification", False)),
                "needs_split": bool(payload.get("needs_split", False)),
                "discard": bool(payload.get("discard", False)),
                "notes": str(payload.get("notes", "")).strip(),
            }
            annotations = load_json(self.annotations_path, {})
            annotations[cluster_id] = annotation
            save_json(self.annotations_path, annotations)
            response_json(self, {"ok": True, "cluster_id": cluster_id, "annotation": annotation, "count": len(annotations)})
        except Exception as e:
            response_json(self, {"ok": False, "error": str(e)}, 400)

    def serve_file(self, path: str) -> None:
        p = Path(path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(path)
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


HTML_PAGE = r"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Anotador de clusters — Projeto 2</title>
  <style>
    :root {
      --bg: #06111f;
      --panel: #ffffff;
      --ink: #07111f;
      --muted: #334155;
      --accent: #1d4ed8;
      --ok: #15803d;
      --warn: #b45309;
      --bad: #b91c1c;
      --focus: #facc15;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      font-size: 22px;
      line-height: 1.45;
    }
    header {
      position: sticky;
      top: 0;
      z-index: 5;
      background: #020617;
      color: white;
      padding: 18px 28px;
      display: flex;
      gap: 24px;
      align-items: center;
      justify-content: space-between;
      border-bottom: 4px solid #2563eb;
    }
    h1 { font-size: 34px; margin: 0; }
    main {
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px;
    }
    .panel {
      background: var(--panel);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 12px 40px rgba(0,0,0,.35);
      margin-bottom: 22px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      align-items: end;
    }
    label { font-weight: 800; display: block; margin-bottom: 8px; }
    input[type="text"], textarea, select {
      width: 100%;
      font-size: 24px;
      padding: 14px 16px;
      border: 3px solid #64748b;
      border-radius: 12px;
      background: #f8fafc;
      color: black;
    }
    textarea { min-height: 110px; resize: vertical; }
    button {
      font-size: 24px;
      font-weight: 800;
      padding: 16px 20px;
      border: 0;
      border-radius: 14px;
      cursor: pointer;
      color: white;
      background: var(--accent);
      min-height: 64px;
    }
    button:focus, input:focus, textarea:focus, select:focus {
      outline: 6px solid var(--focus);
      outline-offset: 2px;
    }
    button.ok { background: var(--ok); }
    button.warn { background: var(--warn); }
    button.bad { background: var(--bad); }
    button.secondary { background: #475569; }
    button.active { box-shadow: 0 0 0 6px var(--focus) inset; }
    .status {
      color: #dbeafe;
      font-weight: 800;
      font-size: 22px;
    }
    .grid {
      display: grid;
      grid-template-columns: 1.1fr .9fr;
      gap: 22px;
    }
    .images {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }
    .images img {
      width: 100%;
      height: 220px;
      object-fit: contain;
      background: #e2e8f0;
      border: 3px solid #0f172a;
      border-radius: 12px;
    }
    .stats {
      font-size: 22px;
      color: var(--muted);
    }
    .stats strong { color: black; }
    .pill {
      display: inline-block;
      background: #dbeafe;
      color: #0f172a;
      padding: 8px 12px;
      border-radius: 999px;
      margin: 4px;
      font-weight: 800;
    }
    .kbd {
      font-family: monospace;
      background: #111827;
      color: white;
      padding: 3px 8px;
      border-radius: 6px;
    }
    .big-id {
      font-size: 42px;
      font-weight: 900;
      color: #0f172a;
    }
    .muted { color: #475569; }
    @media (max-width: 1000px) {
      .grid { grid-template-columns: 1fr; }
      header { display: block; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Anotador de clusters</h1>
    <div class="status" id="status">carregando…</div>
  </header>
  <main>
    <section class="panel controls" aria-label="navegação">
      <button class="secondary" id="prevBtn">← Anterior</button>
      <button class="secondary" id="nextBtn">Próximo →</button>
      <div>
        <label for="jump">Ir para cluster</label>
        <select id="jump"></select>
      </div>
      <div>
        <label for="filter">Filtro</label>
        <select id="filter">
          <option value="all">todos</option>
          <option value="todo">não anotados</option>
          <option value="usable">usáveis</option>
          <option value="split">precisa dividir</option>
          <option value="discard">descartados</option>
        </select>
      </div>
    </section>

    <section class="panel">
      <div class="grid">
        <div>
          <div class="big-id" id="clusterId">—</div>
          <div class="muted" id="clusterCount">—</div>
          <div class="images" id="images"></div>
        </div>
        <div>
          <div class="stats" id="stats"></div>
          <hr />
          <label for="labelFinal">Label final</label>
          <input id="labelFinal" type="text" placeholder="ex: desktop_screenshots" autocomplete="off" />

          <div style="display:grid; grid-template-columns: 1fr; gap: 12px; margin-top: 18px;">
            <button class="ok" id="usableBtn">Usar para classificação</button>
            <button class="warn" id="splitBtn">Precisa dividir</button>
            <button class="bad" id="discardBtn">Descartar</button>
          </div>

          <label for="notes" style="margin-top:20px;">Observações</label>
          <textarea id="notes" placeholder="Notas rápidas sobre o cluster"></textarea>

          <p class="muted">
            Atalhos:
            <span class="kbd">←</span>/<span class="kbd">→</span> navegam,
            <span class="kbd">U</span> usar,
            <span class="kbd">D</span> descartar,
            <span class="kbd">S</span> dividir.
          </p>
        </div>
      </div>
    </section>
  </main>

<script>
let clusters = [];
let annotations = {};
let current = 0;
let saveTimer = null;
let dirty = false;
let saving = false;

const $ = (id) => document.getElementById(id);

function editingTextField() {
  return document.activeElement && ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName);
}

async function fetchState(keepPosition=true) {
  // Não puxa estado remoto enquanto a pessoa está digitando; isso evita cortar observações no meio.
  if (dirty || saving || editingTextField()) return;
  const res = await fetch('/api/state');
  const state = await res.json();
  const currentId = clusters[current]?.id;
  clusters = state.clusters || [];
  annotations = state.annotations || {};
  if (keepPosition && currentId) {
    const idx = clusters.findIndex(c => c.id === currentId);
    if (idx >= 0) current = idx;
  }
  renderJump();
  render();
}

function annFor(c) {
  return annotations[c.id] || {
    label_final: '',
    usable_for_classification: false,
    needs_split: false,
    discard: false,
    notes: ''
  };
}

function visibleClusters() {
  const f = $('filter').value;
  if (f === 'all') return clusters;
  return clusters.filter(c => {
    const a = annFor(c);
    if (f === 'todo') return !annotations[c.id];
    if (f === 'usable') return a.usable_for_classification;
    if (f === 'split') return a.needs_split;
    if (f === 'discard') return a.discard;
    return true;
  });
}

function currentCluster() {
  const list = visibleClusters();
  if (!list.length) return null;
  if (current >= list.length) current = list.length - 1;
  if (current < 0) current = 0;
  return list[current];
}

function renderJump() {
  const sel = $('jump');
  const previous = sel.value;
  sel.innerHTML = '';
  for (const c of visibleClusters()) {
    const opt = document.createElement('option');
    opt.value = c.id;
    opt.textContent = `${c.id} · n=${c.summary?.n ?? '?'}`;
    sel.appendChild(opt);
  }
  if (previous) sel.value = previous;
}

function formatCounts(items) {
  if (!items || !items.length) return '<span class="muted">sem dados</span>';
  return items.map(x => `<span class="pill">${escapeHtml(x.value)} · ${x.count} · ${x.pct}%</span>`).join(' ');
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

function render() {
  const list = visibleClusters();
  const c = currentCluster();
  const annotatedCount = Object.keys(annotations).length;
  $('status').textContent = `${annotatedCount}/${clusters.length} anotados · filtro ${current+1}/${Math.max(1, list.length)} · autosave ${dirty ? 'pendente' : 'ok'}`;
  if (!c) {
    $('clusterId').textContent = 'Nenhum cluster';
    $('images').innerHTML = '';
    return;
  }
  const a = annFor(c);
  $('clusterId').textContent = c.id;
  $('clusterCount').textContent = `${c.summary?.n ?? '?'} imagens · parent ${c.parent_id}`;
  $('jump').value = c.id;
  $('labelFinal').value = a.label_final || '';
  $('notes').value = a.notes || '';
  $('usableBtn').classList.toggle('active', !!a.usable_for_classification);
  $('splitBtn').classList.toggle('active', !!a.needs_split);
  $('discardBtn').classList.toggle('active', !!a.discard);
  $('stats').innerHTML = `
    <p><strong>Pastas principais</strong><br>${formatCounts(c.summary?.top_folders)}</p>
    <p><strong>Aspect ratio</strong><br>${formatCounts(c.summary?.aspect_buckets)}</p>
    <p><strong>Extensões</strong><br>${formatCounts(c.summary?.extensions)}</p>
    <p><strong>Tamanho mediano</strong>: ${c.summary?.median_fsize_kb ?? '?'} KB</p>
    <p><strong>Arquivos exemplo</strong><br>${(c.summary?.sample_files || []).map(escapeHtml).join('<br>')}</p>
  `;
  const reps = c.representative_indices || [];
  $('images').innerHTML = reps.slice(0, 12).map(idx =>
    `<img src="/api/image/${idx}" alt="Imagem representativa ${idx}" loading="lazy">`
  ).join('');
}

function collectAnnotation() {
  const c = currentCluster();
  if (!c) return null;
  return {
    cluster_id: c.id,
    label_final: $('labelFinal').value,
    usable_for_classification: $('usableBtn').classList.contains('active'),
    needs_split: $('splitBtn').classList.contains('active'),
    discard: $('discardBtn').classList.contains('active'),
    notes: $('notes').value
  };
}

async function saveNow() {
  const payload = collectAnnotation();
  if (!payload) return;
  dirty = true;
  saving = true;
  // Atualização otimista local: não chama render() com anotação antiga enquanto o usuário digita.
  annotations[payload.cluster_id] = {
    label_final: payload.label_final,
    usable_for_classification: payload.usable_for_classification,
    needs_split: payload.needs_split,
    discard: payload.discard,
    notes: payload.notes
  };
  const res = await fetch('/api/annotation', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  if (data.ok) {
    annotations[payload.cluster_id] = data.annotation;
    dirty = false;
    saving = false;
    if (!editingTextField()) render(); else document.getElementById('status').textContent = `${Object.keys(annotations).length}/${clusters.length} anotados · autosave ok`;
  } else {
    document.getElementById('status').textContent = 'erro ao salvar: ' + data.error;
    saving = false;
  }
}

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveNow, 1200);
}

function go(delta) {
  current += delta;
  const list = visibleClusters();
  if (current < 0) current = 0;
  if (current >= list.length) current = list.length - 1;
  render();
  window.scrollTo({top: 0, behavior: 'smooth'});
}

$('prevBtn').onclick = () => go(-1);
$('nextBtn').onclick = () => go(1);
$('jump').onchange = () => {
  const list = visibleClusters();
  current = Math.max(0, list.findIndex(c => c.id === $('jump').value));
  render();
};
$('filter').onchange = () => { current = 0; renderJump(); render(); };
$('labelFinal').oninput = scheduleSave;
$('notes').oninput = scheduleSave;
$('usableBtn').onclick = () => { $('usableBtn').classList.toggle('active'); if ($('usableBtn').classList.contains('active')) $('discardBtn').classList.remove('active'); saveNow(); };
$('splitBtn').onclick = () => { $('splitBtn').classList.toggle('active'); saveNow(); };
$('discardBtn').onclick = () => { $('discardBtn').classList.toggle('active'); if ($('discardBtn').classList.contains('active')) $('usableBtn').classList.remove('active'); saveNow(); };

document.addEventListener('keydown', (ev) => {
  if (ev.target.tagName === 'INPUT' || ev.target.tagName === 'TEXTAREA') return;
  if (ev.key === 'ArrowRight') go(1);
  if (ev.key === 'ArrowLeft') go(-1);
  if (ev.key.toLowerCase() === 'u') $('usableBtn').click();
  if (ev.key.toLowerCase() === 'd') $('discardBtn').click();
  if (ev.key.toLowerCase() === 's') $('splitBtn').click();
});

setInterval(() => fetchState(true), 3500);
fetchState(false);
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    hierarchy = load_json(args.hierarchy, {})
    if not hierarchy:
        raise SystemExit(f"Hierarchy not found or empty: {args.hierarchy}. Run hierarchical_kmeans.py first.")
    ClusterAnnotatorHandler.hierarchy_path = args.hierarchy
    ClusterAnnotatorHandler.annotations_path = args.annotations
    ClusterAnnotatorHandler.paths = load_lines(args.paths)
    viz = load_lines(args.viz_paths)
    ClusterAnnotatorHandler.viz_paths = viz if viz else ClusterAnnotatorHandler.paths
    ClusterAnnotatorHandler.leaves = flatten_leaves(hierarchy)
    if not Path(args.annotations).exists():
        save_json(args.annotations, {})
    server = HTTPServer((args.host, args.port), ClusterAnnotatorHandler)
    print(f"Cluster annotator running at http://{args.host}:{args.port}")
    print(f"Clusters: {len(ClusterAnnotatorHandler.leaves)}")
    print(f"Saving annotations to: {args.annotations}")
    server.serve_forever()


if __name__ == "__main__":
    main()
