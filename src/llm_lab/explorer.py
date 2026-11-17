"""The explorer: one self-contained HTML page (no network requests, no weights).

    python -m llm_lab.explorer --serve               # http://127.0.0.1:8767
    python -m llm_lab.explorer --out explorer.html   # write the page to a file

It is built from ``web/`` and shipped as ``assets/explorer.html``. Every dimension it shows comes
from the published configs of GPT-2, Gemma 2B, Qwen2.5-0.5B, DeepSeek-V2-Lite, Qwen3-Next-80B-A3B
and DeepSeek-V3.2 (the same numbers as ``llm_lab.reference.config``).
"""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path

_NEXT_SLOT = '<script type="application/json" id="next-token-data">{}</script>'


def page_tokens(html: str | None = None) -> dict:
    """The page's example inputs: ``{'examples': [{key, label, sentence, candidates, models: {…}}, …]}``,
    each with its tokenization by every model's tokenizer."""
    html = html if html is not None else _template()
    match = re.search(r'window\.LAB_TOKENS = (\{.*?\});</script>', html, re.S)
    if not match:
        raise RuntimeError('Page has no embedded tokenization')
    return json.loads(match.group(1))


def _template() -> str:
    return files('llm_lab').joinpath('assets/explorer.html').read_text(encoding='utf-8')


def render_html(next_token: Path | None = None) -> str:
    """Return the complete, self-contained explorer page.

    With ``next_token`` (a bundle from ``python -m llm_lab.next_token``) the decoding demo
    uses real logits instead of the labeled illustrative ones.
    """
    html = _template()
    if next_token is None:
        return html
    from .next_token import validate
    bundle = json.loads(Path(next_token).read_text(encoding='utf-8'))
    validate(bundle, page_tokens(html))
    data = json.dumps(bundle, ensure_ascii=False, separators=(',', ':'))
    data = data.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    if _NEXT_SLOT not in html:
        raise RuntimeError('Page has no slot for next-token data')
    return html.replace(_NEXT_SLOT, _NEXT_SLOT.replace('>{}<', f'>{data}<'))


def serve(port: int = 8767, host: str = '127.0.0.1', next_token: Path | None = None) -> None:
    if host != '127.0.0.1':
        raise ValueError('The development server binds only to localhost')
    page = render_html(next_token).encode('utf-8')

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split('?')[0] not in ('/', '/index.html'):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(page)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(page)

        def log_message(self, *args):  # keep the terminal quiet
            pass

    with ThreadingHTTPServer((host, port), Handler) as server:
        print(f'LLM Internals Lab: http://{host}:{port}', flush=True)
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description='LLM Internals Lab: serve or export the explorer page')
    parser.add_argument('--serve', action='store_true', help='Run the page on a local web server')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--out', type=Path, default=Path('explorer.html'),
                        help='Write the standalone HTML file here (default when --serve is not given)')
    parser.add_argument('--next-token', type=Path,
                        help='Bundle from `python -m llm_lab.next_token` with real logits for the decoding demo')
    args = parser.parse_args()
    if args.serve:
        serve(port=args.port, next_token=args.next_token)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(render_html(args.next_token), encoding='utf-8')
        print(f'Wrote {args.out.resolve()}')


if __name__ == '__main__':
    main()
