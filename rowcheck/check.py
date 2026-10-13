"""Check one numeric claim against the tables of one or more PDFs.

1. Retrieve the tables whose caption and lines share the most terms with the claim.
2. An open-weight model (Gemma 4 through llama-server) reads the numbered lines and names
   the line, the column header and the cell it relies on, under a JSON schema.
3. Code, not the model, decides the numbers: the cited cell must be a number printed on
   that exact line, and the claimed value is compared with it. The claimed value is also
   searched on every other line, which is how a number lifted from the wrong row is caught.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass, field

from .tables import NUM, Table, norm

ENDPOINT = os.environ.get("ROWCHECK_LLM", "http://127.0.0.1:8471/v1/chat/completions")
TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-]*", re.I)
STOP = set("the a an of on in to and for with is are was were by at as than from its it this that".split())

SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "metric": {"type": "string"},
        "claimed_value": {"type": "string"},
        "conditions": {"type": "array", "items": {"type": "string"}},
        "table": {"type": "string"},
        "row_text": {"type": "string"},
        "line": {"type": "integer"},
        "column": {"type": "string"},
        "cell": {"type": "string"},
        "condition_status": {"type": "array", "items": {"type": "object", "properties": {
            "condition": {"type": "string"},
            "status": {"type": "string", "enum": ["confirmed", "contradicted", "not_stated"]},
            "evidence": {"type": "string"}}, "required": ["condition", "status", "evidence"]}},
        "reason": {"type": "string"},
    },
    "required": ["subject", "metric", "claimed_value", "conditions", "table", "row_text", "line", "column", "cell",
                 "condition_status", "reason"],
}

PROMPT = """You check a numeric claim against tables from a paper.
Claim: {claim}

Tables. Each line is numbered. On labelled lines the row name comes first, then cells as
column=value. A row may wrap onto the next line.
{tables}

Step 1: find the line whose row name is the claim's subject (same model, same size) and the
column that is the claim's metric. Put the table id (T1, T2, ...) into table, copy that line's text into row_text and its
number into line.
Copy the cell value exactly as printed into cell, even when it differs from the claimed value.
Step 2: list every setting the claim states beyond subject and metric (shots, split, fine-tuned
or base, prompt, decoding). For each, status is confirmed if the caption, header or row says the
same, contradicted if they say otherwise, not_stated if the table says nothing about it. Quote
the words you used as evidence. Model size belongs to the subject, not to the settings.
Only when no line has the subject AND a column with the metric, set line to -1 and cell to "".
Answer in JSON."""


@dataclass
class Verdict:
    claim: str
    verdict: str  # supported | contradicted | unsupported | ambiguous
    summary: str
    doc: str = ""
    page: int = 0
    table: str = ""
    line: int = -1
    line_text: str = ""
    column: str = ""
    cell: str = ""
    claimed: str = ""
    elsewhere: list = field(default_factory=list)  # [(table label, page, line no, line text)]
    model: dict = field(default_factory=dict)
    focus: float = 0.0  # horizontal position of the marked cell in the page crop, 0..1

    def to_json(self):
        return asdict(self)


def _terms(s: str) -> list[str]:
    return [t.lower().rstrip(".") for t in TOKEN.findall(s) if t.lower() not in STOP]


def retrieve(claim: str, tables: list[Table], k: int = 3) -> list[Table]:
    q = Counter(_terms(claim))
    df = Counter()
    docs = []
    for t in tables:
        terms = set(_terms(t.caption + " " + " ".join(l.text for l in t.lines)))
        docs.append(terms)
        df.update(terms)
    n = len(tables)
    scores = []
    for t, terms in zip(tables, docs):
        s = sum(math.log(1 + n / df[w]) for w in q if w in terms)
        scores.append(s)
    order = sorted(range(n), key=lambda i: -scores[i])
    return [tables[i] for i in order[:k]]


def _ask(claim: str, cands: list[Table]) -> dict:
    blocks = "\n\n".join(f"[T{i + 1}] {t.label}, page {t.page}: {t.caption}\n{t.numbered()}" for i, t in enumerate(cands))
    body = {
        "messages": [{"role": "user", "content": PROMPT.format(claim=claim, tables=blocks)}],
        "temperature": 0,
        "max_tokens": 1500,
        "chat_template_kwargs": {"enable_thinking": os.environ.get("ROWCHECK_THINK") == "1"},
        "response_format": {"type": "json_schema", "json_schema": {"name": "check", "schema": SCHEMA}},
    }
    req = urllib.request.Request(ENDPOINT, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    text = out["choices"][0]["message"]["content"] or ""
    text = text[text.find("{"): text.rfind("}") + 1]
    return json.loads(text)


def _num(s: str) -> float | None:
    m = re.search(r"[-−]?\d+(?:[.,]\d+)?", s or "")
    return float(norm(m.group(0))) if m else None


SCALE = {"k": "K", "thousand": "K", "m": "M", "million": "M", "b": "B", "billion": "B", "t": "T", "trillion": "T"}
SCALED = re.compile(r"\d+(?:[.,]\d+)?\s*(k|m|b|t|thousand|million|billion|trillion)\b", re.I)


def _scale(text: str, value: float | None) -> str | None:
    """Scale word right after the claimed number, never after a model size like '7B'."""
    for m in SCALED.finditer(text):
        n = _num(m.group(0))
        if value is not None and n is not None and _same(n, value):
            return SCALE[m.group(1).lower()]
    return None


def _table_scale(table: Table, column: str) -> str | None:
    """Scale written in the column header or caption, e.g. '(B tokens)'."""
    for text in (column, " ".join(l.text for l in table._header_lines()), table.caption):
        m = re.search(r"\(\s*(K|M|B|T)\b[^)]*\)|\b(thousand|million|billion|trillion)s?\b", text or "", re.I)
        if m:
            return SCALE[(m.group(1) or m.group(2)).lower()]
    return None


SHOT = re.compile(r"\b(\d+|zero|one|few)[- ]shot", re.I)
WORDS = {"zero": "0", "one": "1"}


def _shots(text: str) -> set[str]:
    return {WORDS.get(m.group(1).lower(), m.group(1).lower()) for m in SHOT.finditer(text or "")}


def _row_context(table: Table, line: int, column: str) -> str:
    """Text that can state a setting for this cell: row label, the next line when it is a
    parenthetical such as '(5-shot)', the column header and the caption."""
    nxt = table.lines[line + 1].text if line + 1 < len(table.lines) else ""
    nxt = nxt if nxt.strip().startswith("(") else ""
    prv = table.lines[line - 1].text if line > 0 else ""
    prv = prv if len(table.lines[line - 1].numbers()) == 0 and len(prv.split()) <= 4 else ""
    return " ".join([prv, table.annotate(line).split(" | ")[0], nxt, column or "", table.caption])


def _same(a: float, b: float) -> bool:
    """Equal at printed precision: 68.6 is not 68.4, and 0.0 matches 0."""
    return abs(a - b) <= max(0.051, 0.0005 * abs(b))


def _label(table: Table, i: int) -> str:
    text = table.annotate(i)
    return text.split(" | ")[0].strip() if " | " in text else ""


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9+]", "", s.lower())


def _resolve_line(table: Table, m: dict) -> int:
    """The model's line, checked against the row text it copied, then against the subject:
    a row named exactly like the subject wins over a longer name ("TÜLU 2" vs "TÜLU 2+DPO")."""
    line = m.get("line", -1)
    row = re.sub(r"^L\d+:\s*", "", m.get("row_text", "") or "").strip()
    if row:
        exact = [i for i in range(len(table.lines)) if table.annotate(i).strip() == row]
        named = [i for i in range(len(table.lines)) if _label(table, i) and
                 _key(_label(table, i)) == _key(row.split(" | ")[0])]
        if exact:
            line = exact[0]
        elif len(named) == 1:
            line = named[0]
    if line is None or not 0 <= line < len(table.lines):
        return -1
    subject = _key(m.get("subject", ""))
    if subject and _key(_label(table, line)) != subject:
        same = [i for i in range(len(table.lines)) if _key(_label(table, i)) == subject]
        if len(same) == 1:
            old, line = table.lines[line], same[0]
            # carry the column over: the cell under the same header on the corrected row
            src = next((w for w in old.words if NUM.match(w.text) and norm(w.text) == norm(m.get("cell", ""))), None)
            if src is not None:
                col = table.column_of(src)
                hit = next((w for w in table.lines[line].words if NUM.match(w.text) and table.column_of(w) == col), None)
                if hit is not None:
                    m["cell"] = hit.text
    return line


def _toks(s: str) -> set[str]:
    return {_key(t) for t in _terms(s or "")} - {""}


def _cell_terms(table: Table, i: int, word) -> set[str]:
    """Words that name one cell: its row label (or a short label line above it), a
    parenthetical on the next line such as '(5-shot)', and its column header."""
    prv = table.lines[i - 1].text if i > 0 and not table.lines[i - 1].numbers() and \
        len(table.lines[i - 1].text.split()) <= 4 else ""
    nxt = table.lines[i + 1].text if i + 1 < len(table.lines) and table.lines[i + 1].text.strip().startswith("(") else ""
    label = _label(table, i) or " ".join(w.text for w in table.lines[i].words if not NUM.match(w.text))
    return _toks(" ".join([prv, label, nxt, table.column_of(word)]))


AVERAGE = re.compile(r"^(avg|average|overall|mean)$", re.I)


def _better_column(claim: str, table: Table, line: int, cell: str):
    """The cell on `line` whose column header matches the claim best, when that is not the
    cell the model cited: a transposed table puts the subject ("Gemma 2B") in the header.
    With no header matching, an unqualified score is the Average column's."""
    said = _toks(claim)
    nums = [w for w in table.lines[line].words if NUM.match(w.text) and table.column_of(w) not in ("", "#label")]
    cited = next((w for w in nums if _num(w.text) is not None and _same(_num(w.text), _num(cell) or math.inf)), None)
    if cited is None or len(nums) < 2:
        return None
    score = {id(w): len(_toks(table.column_of(w)) & said) for w in nums}
    best = max(nums, key=lambda w: score[id(w)])
    if score[id(best)] > score[id(cited)] and sum(score[id(w)] == score[id(best)] for w in nums) == 1:
        return best
    if score[id(best)] == 0 and not AVERAGE.match(table.column_of(cited)):
        avg = [w for w in nums if AVERAGE.match(table.column_of(w))]
        return avg[0] if len(avg) == 1 else None
    return None


def _setting_elsewhere(claim: str, cands: list[Table], shots: set[str], base: int):
    """A cell in another candidate table that states the claim's shot count and names the
    claim's subject and metric at least as well as the model's pick (`base` shared words)."""
    said = _toks(claim)
    best, top = None, base
    for t in cands:
        for i, ln in enumerate(t.lines):
            for w in ln.words:
                if not NUM.match(w.text) or t.column_of(w) in ("", "#label"):
                    continue
                prv = t.lines[i - 1].text if i > 0 else ""
                nxt = t.lines[i + 1].text if i + 1 < len(t.lines) else ""
                if not shots & _shots(" ".join([prv, ln.text, nxt, t.column_of(w)])):
                    continue
                sc = len(_cell_terms(t, i, w) & said)
                if sc > top:
                    best, top = (t, i, w), sc
    return best


def check(claim: str, tables: list[Table], k: int = 3) -> Verdict:
    cands = retrieve(claim, tables, k)
    if not cands:
        return Verdict(claim, "unsupported", "No table in the document.")
    m = _ask(claim, cands)
    claimed = _num(m.get("claimed_value", ""))
    tid = re.search(r"T(\d+)", m.get("table", "") or "")
    table = cands[int(tid.group(1)) - 1] if tid and 0 < int(tid.group(1)) <= len(cands) else cands[0]
    m["cell"] = (m.get("cell") or "").split("=")[-1].strip()
    line = _resolve_line(table, m)
    m["line"] = line
    v = Verdict(claim, "ambiguous", "", doc=table.doc, page=table.page, table=table.label, model=m,
                claimed=m.get("claimed_value", ""), column=re.sub(r"\s*[↑↓]", "", m.get("column", "")).strip())

    # every place the claimed number is printed, across the candidate tables
    if claimed is not None:
        for t in cands:
            for i, ln in enumerate(t.lines):
                if any(_same(float(x), claimed) for x in ln.numbers() if _num(x) is not None):
                    v.elsewhere.append((t.label, t.page, i, ln.text))

    line = m.get("line", -1)
    if line is None or line < 0 or line >= len(table.lines):
        v.verdict = "unsupported"
        v.summary = "No table row measures this claim."
        return v
    if not (m.get("cell") or "").strip():
        v.verdict = "unsupported"
        v.summary = "No table row measures this claim."
        return v
    claim_shots = _shots(claim)
    if claim_shots and not _shots(_row_context(table, line, v.column)):
        src = next((w for w in table.lines[line].words if NUM.match(w.text) and _num(w.text) is not None
                    and _same(_num(w.text), _num(m["cell"]) or math.inf)), None)
        base = len(_cell_terms(table, line, src) & _toks(claim)) if src is not None else 0
        hit = _setting_elsewhere(claim, [t for t in cands if t is not table], claim_shots, base)
        if hit:
            table, line, w = hit
            m["cell"], v.column = w.text, table.column_of(w)
            v.doc, v.page, v.table = table.doc, table.page, table.label
    alt = _better_column(claim, table, line, m.get("cell", ""))
    if alt is not None:
        m["cell"], v.column = alt.text, table.column_of(alt)
    ln = table.lines[line]
    v.line, v.line_text = line, ln.text
    cell = _num(m.get("cell", ""))
    printed = [float(x) for x in ln.numbers() if _num(x) is not None]
    verbatim = next((w.text for w in ln.words if w.text == (m.get("cell") or "").strip()), None)
    if cell is not None and verbatim and not any(_same(cell, p) for p in printed):
        printed.append(cell)  # a cell printed with its unit, such as "13B"
    if cell is None or not any(_same(cell, p) for p in printed):
        # the model's cell is not on its line: do not trust the reading
        v.verdict = "ambiguous"
        v.summary = f"The cited cell {m.get('cell')!r} is not printed on {table.label} line {line}."
        return v
    v.cell = m.get("cell", "")
    if claimed is None:
        v.verdict = "ambiguous"
        v.summary = "The claim states no number."
        return v
    others = [e for e in v.elsewhere if not (e[0] == table.label and e[2] == line)]
    said = set(_terms(claim))
    # a setting counts only if the claim itself says it: models tend to add "base" or "pretrained"
    status = {c.get("condition", ""): c.get("status") for c in m.get("condition_status", [])
              if set(_terms(c.get("condition", ""))) & said}
    bad = [c for c, st in status.items() if st == "contradicted"]
    unstated = [c for c, st in status.items() if st == "not_stated"]
    where = f"{table.label} (p. {table.page})"
    col = v.column if v.column and _key(v.column) != _key(m.get("subject", "")) else m.get("metric")
    what = f"{m.get('subject')} / {col}"
    claim_shots, table_shots = _shots(claim), _shots(_row_context(table, line, v.column))
    if claim_shots and table_shots and not claim_shots & table_shots:
        bad = [b for b in bad if "shot" not in b.lower()] + \
            [f"{'/'.join(sorted(claim_shots))}-shot (the table says {'/'.join(sorted(table_shots))}-shot)"]
        unstated = [u for u in unstated if "shot" not in u.lower()]
    cs, ts = _scale(claim, claimed), _table_scale(table, v.column)
    unit = re.fullmatch(r"[\d.,]+\s*([KMBT])", v.cell.strip(), re.I)
    ts = ts or (unit.group(1).upper() if unit else None)
    if cs and not ts:  # an appendix copy may drop the unit: find the same value where it is stated
        said_words = {w for w in _terms(claim) if not re.fullmatch(r"[\d.]+[a-z]?", w)}
        for t2 in cands:
            if t2 is table:
                continue
            for i2, l2 in enumerate(t2.lines):
                w2 = next((w for w in l2.words if NUM.match(w.text) and _num(w.text) is not None
                           and _same(_num(w.text), cell)), None)
                if w2 is None:
                    continue
                ctx = set(_terms(_label(t2, i2) + " " + t2.column_of(w2)))
                s2 = _table_scale(t2, t2.column_of(w2))
                if s2 and len(ctx & said_words) >= 2:
                    ts, where = s2, f"{t2.label} (p. {t2.page})"
                    break
            if ts:
                break
    if _same(claimed, cell) and cs and ts and cs != ts:
        v.verdict = "contradicted"
        v.summary = f"{where} reports {v.cell} in {ts} units for {what}; the claim says {cs}."
    elif _same(claimed, cell) and cs and not ts:
        v.verdict = "ambiguous"
        v.summary = f"{where} reports {v.cell} for {what} but gives no unit; the claim says {cs}."
    elif not _same(claimed, cell):
        v.verdict = "contradicted"
        v.summary = f"{where} reports {v.cell}, not {v.claimed}, for {what}."
        same_row = [w for w in ln.words if NUM.match(w.text) and _num(w.text) is not None
                    and _same(_num(w.text), claimed)]
        if same_row:
            col = table.column_of(same_row[0])
            v.summary += f" The same row prints {same_row[0].text} under {col or 'another column'}."
        elif others:
            t, p, i, txt = others[0]
            tab = next((x for x in cands if x.label == t and x.page == p), None)
            label = _label(tab, i) if tab else ""
            word = next((w for w in tab.lines[i].words if NUM.match(w.text) and _num(w.text) is not None
                         and _same(_num(w.text), claimed)), None) if tab else None
            col = tab.column_of(word) if word is not None else ""
            if label and col and col != "#label":
                near = " (the next row)" if tab is table and i == line + 1 else \
                    " (the row above)" if tab is table and i == line - 1 else ""
                v.summary += f" {word.text} is {label}'s {col}{near}, {t} p. {p}."
            else:
                v.summary += f" {v.claimed} is printed on {t} line {i} (p. {p}): \"{txt[:90]}\"."
    elif bad:
        v.verdict = "contradicted"
        v.summary = f"{where} reports {v.cell} for {what}, but not under {', '.join(bad)}."
    elif unstated:
        v.verdict = "ambiguous"
        v.summary = f"{where} reports {v.cell} for {what}; the table does not state {', '.join(unstated)}."
    else:
        v.verdict = "supported"
        v.summary = f"{where} reports {v.cell} for {what}."
    return v
