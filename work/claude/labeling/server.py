"""Local plate-labelling server (stdlib only). Frames never leave this machine.
Run:  python3 work/claude/labeling/server.py   then open http://localhost:8765
Labels are written to work/claude/labeling/labels/<frame>.json (human labels, original pixel coords)."""
import json, re, sys, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAMES, SUGG, LABELS = HERE / 'frames', HERE / 'suggestions', HERE / 'labels'
LABELS.mkdir(exist_ok=True)
MANIFEST = json.loads((HERE / 'manifest.json').read_text())
ORDER = [f'frame_{t:04d}' for t in MANIFEST['train_times'] + MANIFEST['locked_test_times']]
SPLIT = {f'frame_{t:04d}': 'train' for t in MANIFEST['train_times']}
SPLIT.update({f'frame_{t:04d}': 'locked_test' for t in MANIFEST['locked_test_times']})
STEM = re.compile(r'^frame_\d{4}$')


def frame_list():
    out = []
    for s in ORDER:
        lab = LABELS / f'{s}.json'
        d = json.loads(lab.read_text()) if lab.exists() else None
        out.append(dict(stem=s, split=SPLIT[s], status=d['status'] if d else 'todo',
                        plates=len(d['plates']) if d else None,
                        cng=sum(p['vehicle'] == 'cng' for p in d['plates']) if d else None,
                        suggestions_ready=(SUGG / f'{s}.json').exists()))
    return out


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype='application/json'):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.split('?')[0]
        if p == '/':
            return self.send(200, (HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        if p == '/api/frames':
            return self.send(200, frame_list())
        m = re.match(r'^/api/frame/(frame_\d{4})$', p)
        if m and m[1] in SPLIT:
            s = m[1]
            lab, sug = LABELS / f'{s}.json', SUGG / f'{s}.json'
            return self.send(200, dict(stem=s, split=SPLIT[s],
                                       labels=json.loads(lab.read_text()) if lab.exists() else None,
                                       suggestions=json.loads(sug.read_text()) if sug.exists() else None))
        m = re.match(r'^/img/(frame_\d{4})\.jpg$', p)
        if m and m[1] in SPLIT:
            return self.send(200, (FRAMES / f'{m[1]}.jpg').read_bytes(), 'image/jpeg')
        self.send(404, {'error': 'not found'})

    def do_POST(self):
        m = re.match(r'^/api/frame/(frame_\d{4})$', self.path)
        if not (m and m[1] in SPLIT):
            return self.send(404, {'error': 'not found'})
        d = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        def box(b):
            x1, y1, x2, y2 = (int(round(float(v))) for v in b)
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        kinds = {'car', 'motorcycle', 'cng', 'leguna', 'bus_truck', 'other'}
        rec = dict(file=f'{m[1]}.jpg', split=SPLIT[m[1]],
                   status=d.get('status', 'done') if d.get('status') in ('done', 'skipped') else 'done',
                   labeler='human (labelling UI)', saved_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
                   plates=[dict(box=box(p['box']), vehicle=p['vehicle'] if p['vehicle'] in kinds else 'other',
                                from_suggestion=bool(p.get('from_suggestion'))) for p in d.get('plates', [])],
                   ignore=[dict(box=box(p['box'])) for p in d.get('ignore', [])],
                   notes=str(d.get('notes', ''))[:2000])
        (LABELS / f'{m[1]}.json').write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        self.send(200, {'ok': True, 'plates': len(rec['plates'])})


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f'Labelling UI: http://localhost:{port}  ({len(ORDER)} frames)', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), H).serve_forever()
