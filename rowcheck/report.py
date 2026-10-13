"""Page crops with the checked row marked, and a self-contained HTML report."""

from __future__ import annotations

import base64
import html
import re

import pymupdf

from .check import Verdict, _num, _same
from .tables import NUM, Table, _is_data, norm

COLORS = {"supported": (0.13, 0.55, 0.27), "contradicted": (0.8, 0.15, 0.1),
          "ambiguous": (0.85, 0.55, 0.0), "unsupported": (0.45, 0.45, 0.45)}


def crop_png(pdf: str, table: Table, v: Verdict, zoom: float = 2.5) -> bytes:
    doc = pymupdf.open(pdf)
    page = doc[table.page - 1]
    cell_x = None
    color = COLORS[v.verdict]
    if 0 <= v.line < len(table.lines):
        ln = table.lines[v.line]
        x0, y0, x1, y1 = ln.bbox
        page.draw_rect(pymupdf.Rect(x0 - 2, y0 - 1, x1 + 2, y1 + 1), color=color, width=1.2)
        for w in ln.words:  # the cited cell
            if NUM.match(w.text) and v.cell and norm(w.text) == norm(v.cell):
                cell_x = (w.bbox[0] + w.bbox[2]) / 2
                page.draw_rect(pymupdf.Rect(w.bbox).include_point((w.bbox[0] - 1, w.bbox[1] - 1)),
                               color=color, fill=color, fill_opacity=0.25, width=0.8)
                break
    target = _num(v.claimed)
    cell = _num(v.cell)
    if v.verdict == "contradicted" and target is not None and (cell is None or not _same(target, cell)):
        for label, pno, i, _ in v.elsewhere:
            if label == table.label and pno == table.page:
                for w in table.lines[i].words:
                    if NUM.match(w.text) and _num(w.text) is not None and _same(_num(w.text), target):
                        page.draw_rect(pymupdf.Rect(w.bbox), color=(0.1, 0.3, 0.8), width=1.0, dashes="[2] 0")
    hdr = table._header_lines()
    start = table.lines.index(hdr[0]) if hdr else table._first_data()
    data = [i for i, l in enumerate(table.lines) if i >= start and _is_data(l)]
    body = table.lines[start:(data[-1] + 1) if data else None]
    x0 = min(l.bbox[0] for l in body); y0 = min(l.bbox[1] for l in body)
    x1 = max(l.bbox[2] for l in body); y1 = max(l.bbox[3] for l in body)
    clip = pymupdf.Rect(x0 - 12, y0 - 6, x1 + 12, y1 + 6) & page.rect
    if cell_x is not None:
        v.focus = round((cell_x - clip.x0) / max(1.0, clip.width), 3)
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip).tobytes("png")


CSS = """:root{--bg:#fbfaf7;--fg:#1d1d1b;--mute:#6b6a65;--card:#fff;--line:#e4e1da;
--supported:#1f7a3f;--contradicted:#b42318;--ambiguous:#b26b00;--unsupported:#6b6a65}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--mute:#a19f98;--card:#1f1f1c;--line:#34332f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:880px;margin:0 auto;padding:40px 16px 64px}h1{font-size:26px;line-height:1.25;margin:0 0 28px;font-weight:650}
.card{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--c);border-radius:10px;padding:18px 20px;margin:0 0 20px}
.v{display:inline-block;font:600 12px/1 ui-monospace,monospace;letter-spacing:.08em;text-transform:uppercase;color:var(--c)}
.claim{font-size:19px;font-weight:600;margin:8px 0 6px}.sum{margin:0 0 6px}.src{color:var(--mute);font-size:13px}
.src a{color:inherit}.crop{overflow-x:auto;margin-top:14px;border-radius:6px;border:1px solid var(--line);background:#fff}
.crop img{display:block;width:100%;min-width:640px}
.supported{--c:var(--supported)}.contradicted{--c:var(--contradicted)}.ambiguous{--c:var(--ambiguous)}.unsupported{--c:var(--unsupported)}"""


FOCUS = """function f(){document.querySelectorAll('.crop').forEach(c=>{const x=parseFloat(c.dataset.focus)||0;
c.scrollLeft=Math.max(0,x*c.scrollWidth-c.clientWidth/2)})}
document.querySelectorAll('.crop img').forEach(i=>i.complete?f():i.addEventListener('load',f));addEventListener('resize',f)"""


def html_report(items: list[tuple[Verdict, bytes | None]], title: str) -> str:
    cards = []
    for v, png in items:
        img = (f'<div class="crop" data-focus="{v.focus}"><img alt="{html.escape(v.table)}, page {v.page}, checked row marked" '
               f'src="data:image/png;base64,{base64.b64encode(png).decode()}"></div>') if png else ""
        doc = html.escape(v.doc)
        link = f'<a href="https://arxiv.org/abs/{doc}">arXiv {doc}</a>' if re.fullmatch(r"\d{4}\.\d{4,5}", v.doc) else doc
        src = f"{link} · {html.escape(v.table)} · p. {v.page}" if v.line >= 0 else link
        cards.append(f'<section class="card {v.verdict}"><span class="v">{v.verdict}</span>'
                     f'<p class="claim">{html.escape(v.claim)}</p><p class="sum">{html.escape(v.summary)}</p>'
                     f'<div class="src">{src}</div>{img}</section>')
    return (f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{CSS}</style><main><h1>{html.escape(title)}</h1>{"".join(cards)}</main>'
            f'<script>{FOCUS}</script></html>')
