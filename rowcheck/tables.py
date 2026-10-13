"""Table regions out of a PDF, kept as positioned words.

Paper tables are mostly borderless, with merged headers and multi-line cells, so grid
reconstruction fails often. Instead each table is a list of visual lines, each line a list
of words with their boxes. The model reads the lines by number; every value it cites is
then checked against the words on that exact line, and the page box comes from those words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

CAPTION = re.compile(r"^\s*(Table|Tab\.)\s*([0-9]+|[IVX]+)\s*[:.|]?", re.I)
SIZE = re.compile(r"\d+(?:\.\d+)?[BMKT]", re.I)
SENTENCE = re.compile(r"[a-z]{3,}[.,;:](\s|$)")
NUM = re.compile(r"^[-−+]?\(?\d+(?:[.,]\d+)?%?\)?[*∗†‡]?$")


@dataclass
class Word:
    text: str
    bbox: tuple[float, float, float, float]


@dataclass
class Line:
    words: list[Word]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def bbox(self):
        xs0, ys0, xs1, ys1 = zip(*(w.bbox for w in self.words))
        return (min(xs0), min(ys0), max(xs1), max(ys1))

    def numbers(self) -> list[str]:
        return [norm(w.text) for w in self.words if NUM.match(w.text)]


@dataclass
class Table:
    doc: str
    page: int  # 1-based
    label: str  # "Table 3"
    caption: str
    lines: list[Line]

    @property
    def bbox(self):
        xs0, ys0, xs1, ys1 = zip(*(l.bbox for l in self.lines))
        return (min(xs0), min(ys0), max(xs1), max(ys1))

    def numbered(self) -> str:
        return "\n".join(f"L{i:02d}: {self.annotate(i)}" for i in range(len(self.lines)))

    def _first_data(self) -> int:
        for i, l in enumerate(self.lines):
            if _is_data(l):
                return i
        return 0

    def _header_lines(self) -> list[Line]:
        """Non-data lines directly above the first data line, walking up while the vertical
        gap stays table-like. A line below the widest one that starts in the row-label
        column is a section row, not a header."""
        first = self._first_data()
        if first == 0:
            return []
        hts = sorted(l.bbox[3] - l.bbox[1] for l in self.lines[first:] if _is_data(l)) or [10]
        step = hts[len(hts) // 2] * 1.9
        cands, k, skipped = [], first - 1, 0
        below = self.lines[first]
        while k >= 0 and len(cands) < 3:
            l = self.lines[k]
            if _is_data(l):
                break
            lower = sum(bool(re.fullmatch(r"[a-z][a-z\-]{2,}[,.;:)]*", w.text)) for w in l.words)
            if _is_prose(l) or (len(l.words) >= 6 and (SENTENCE.search(l.text) or lower >= 4)) \
                    or not self._aligned(l):
                skipped += 1  # caption text from the other column can sit between header and rows
                if skipped > 3:
                    break
                k -= 1
                continue
            if below.bbox[1] - l.bbox[3] > step * (1 + skipped):
                break
            cands.insert(0, (k, l))
            below, k = l, k - 1
        if not cands:
            return []
        # main header: the line that starts furthest left (it names the row-label column)
        k, primary = min(cands, key=lambda kl: (round(kl[1].words[0].bbox[0] / 4), -len(kl[1].words)))
        edge = primary.words[0].bbox[2]
        return [l for kk, l in cands if kk <= k or (l.words[0].bbox[0] > edge + 2 and
                                                     (_gap(l) > 8 or l.text.lstrip().startswith("(")))]

    def _centres(self) -> list[float]:
        if hasattr(self, "_cen"):
            return self._cen
        first = self._first_data()
        pts = sorted((w.bbox[0] + w.bbox[2]) / 2 for l in self.lines[first:]
                     if _is_data(l) for w in l.words[_lead(l):] if NUM.match(w.text))
        groups: list[list[float]] = []
        for c in pts:
            if groups and c - groups[-1][-1] <= 9:
                groups[-1].append(c)
            else:
                groups.append([c])
        nrows = max(1, sum(_is_data(l) for l in self.lines[first:]))
        self._cen = [sum(g) / len(g) for g in groups if len(g) >= max(2, nrows // 3)]
        return self._cen

    def _aligned(self, l: Line) -> bool:
        cen = self._centres()
        if not cen:
            return True
        spacing = min((b - a for a, b in zip(cen, cen[1:])), default=60)
        right = [h for h in l.words if h.bbox[2] > cen[0] - spacing / 2]
        if not right:
            return True  # only a row-label header word
        ok = sum(min(abs(c - (h.bbox[0] + h.bbox[2]) / 2) for c in cen) <= spacing / 2 + 2
                 or h.bbox[2] - h.bbox[0] > spacing * 1.3 for h in right)
        return ok >= 0.6 * len(right)

    def _columns(self) -> list[tuple[float, str]]:
        """Column centres from where numbers line up across the data rows, each with the
        header words stacked above it (nearest centre, within half the column spacing)."""
        if hasattr(self, "_cols"):
            return self._cols
        first = self._first_data()
        centres = sorted((w.bbox[0] + w.bbox[2]) / 2 for l in self.lines[first:]
                         if _is_data(l) for w in l.words[_lead(l):] if NUM.match(w.text))
        groups: list[list[float]] = []
        for c in centres:
            if groups and c - groups[-1][-1] <= 9:
                groups[-1].append(c)
            else:
                groups.append([c])
        nrows = max(1, sum(_is_data(l) for l in self.lines[first:]))
        digits = {}
        for l in self.lines[first:]:
            if _is_data(l):
                for w in l.words[_lead(l):]:
                    if NUM.match(w.text):
                        digits.setdefault(round((w.bbox[0] + w.bbox[2]) / 2), []).append(w.text)
        def label_like(g):  # a column of single digits ("LLaMA 2") belongs to the row label
            vals = [v for c in g for v in digits.get(round(c), [])]
            return vals and all(re.fullmatch(r"\d", v) for v in vals)
        cols = [sum(g) / len(g) for g in groups if len(g) >= max(2, nrows // 3) and not label_like(g)]
        names = {c: [] for c in cols}
        if cols:
            spacing = min((b - a for a, b in zip(cols, cols[1:])), default=60)
            heads = self._header_lines()
            for hi, l in enumerate(heads):
                runs = _runs([h for h in l.words if (h.bbox[0] + h.bbox[2]) / 2 > cols[0] - spacing / 2])
                below = heads[hi + 1].words if hi < len(heads) - 1 else []
                named_below = sum(any(abs(c - (h.bbox[0] + h.bbox[2]) / 2) <= spacing / 2 + 2 for h in below) for c in cols)
                owner = {c: min(runs, key=lambda r: abs(r[0] - c)) for c in cols} if runs else {}
                centred = all(abs(r[0] - sum(mine) / len(mine)) <= spacing / 3
                              for r in runs for mine in [[c for c in cols if owner[c] is r]] if mine)
                if 2 <= len(runs) < len(cols) and named_below >= 0.8 * len(cols) and centred:
                    # a group header centred over the column names it spans ("Gemma" over 2B, 7B)
                    for c in cols:
                        names[c].append(owner[c][1])
                    continue
                near_ok = sum(abs(min(abs(c - (h.bbox[0] + h.bbox[2]) / 2) for c in cols)) <= spacing / 2 + 2
                              for h in l.words if h.bbox[0] > cols[0] - spacing)
                right = [h for h in l.words if h.bbox[0] > cols[0] - spacing]
                if right and near_ok < 0.5 * len(right):
                    continue  # running head or prose, not aligned with the columns
                for h in l.words:
                    hc = (h.bbox[0] + h.bbox[2]) / 2
                    near = min(cols, key=lambda c: abs(c - hc))
                    wide = h.bbox[2] - h.bbox[0] > spacing * 1.3  # a group label spanning columns
                    if abs(near - hc) <= spacing / 2 + 2 or (wide and h.bbox[0] - 2 <= near <= h.bbox[2] + 2):
                        names[near].append(h.text)
        self._cols = [(c, " ".join(names[c])) for c in cols]
        return self._cols

    def column_of(self, word: Word) -> str:
        cols = self._columns()
        if not cols:
            return ""
        xc = (word.bbox[0] + word.bbox[2]) / 2
        c, name = min(cols, key=lambda cn: abs(cn[0] - xc))
        if abs(c - xc) > 9:
            return "#label" if xc < cols[0][0] else ""
        return name or f"col{[x for x, _ in cols].index(c) + 1}"

    def _families(self) -> dict[int, str]:
        """Model family names printed once, vertically centred over their size rows:
        "MPT" between "7B" and "30B". Each size-only row gets the family whose label sits
        symmetrically around it; a row closer to another family's label stops the group.
        Standalone labels claim rows first, a family merged into a row ("QWEN 7B") second."""
        if hasattr(self, "_fam"):
            return self._fam
        first, n = self._first_data(), len(self.lines)
        def size_only(i):
            l = self.lines[i]
            return _is_data(l) and SIZE.fullmatch(l.words[0].text) is not None and NUM.match(l.words[1].text) is not None
        cols = self._centres()
        alone, merged = {}, {}
        for i in range(first, n):
            l = self.lines[i]
            if not _is_data(l) and 1 <= len(l.words) <= 3 and l.words[0].text[:1].isalpha() and \
                    (not cols or l.bbox[2] < cols[0]) and not l.text.strip().startswith("("):
                alone[i] = l.text
            elif _is_data(l):
                k = next((j for j, w in enumerate(l.words) if NUM.match(w.text)), 0)
                if k >= 2 and SIZE.fullmatch(l.words[k - 1].text):
                    merged[i] = " ".join(w.text for w in l.words[:k - 1])
        fam: dict[int, str] = {}
        for labels in (alone, merged):
            for p, name in labels.items():
                others = [q for q in (alone if labels is alone else list(alone) + list(merged)) if q != p]
                for j in range(1, n):
                    up, dn = p - j, p + j
                    if up < first or dn >= n or not (size_only(up) and size_only(dn)) or up in fam or dn in fam:
                        break
                    if any(abs(q - r) < j for r in (up, dn) for q in others):
                        break
                    fam[up] = fam[dn] = name
        self._fam = fam
        return fam

    def annotate(self, i: int) -> str:
        line = self.lines[i]
        if i < self._first_data() or not _is_data(line):
            return line.text
        lead = _lead(line)
        nums = [w for w in line.words[lead:] if NUM.match(w.text)]
        cols = [self.column_of(w) for w in nums]
        data = [c for c in cols if c != "#label"]
        if not data or "" in data or len(set(data)) != len(data):
            return line.text  # labels not one-to-one: give the raw line, never a wrong label
        parts, label = [], [w.text for w in line.words[:lead]]
        for w in line.words[lead:]:
            col = self.column_of(w) if NUM.match(w.text) else ""
            if not parts and (col == "#label" or not NUM.match(w.text)):
                label.append(w.text)
            elif NUM.match(w.text):
                parts.append(f"{col}={w.text}" if col and col != "#label" else w.text)
            else:
                parts.append(w.text)
        if i in self._families():
            label.insert(0, self._families()[i])
        return " ".join(label) + " | " + " | ".join(parts)


def norm(s: str) -> str:
    return s.replace("−", "-").strip("()*∗†‡%+").replace(",", ".")


def _lines(page) -> list[Line]:
    """Words grouped into visual lines: same baseline within 2.5 pt, split at wide gaps
    only when the page has two text columns (a gap crossing the page middle)."""
    words = [Word(w[4], tuple(w[:4])) for w in page.get_text("words")]
    words.sort(key=lambda w: ((w.bbox[1] + w.bbox[3]) / 2, w.bbox[0]))
    out: list[list[Word]] = []
    for w in words:
        yc = (w.bbox[1] + w.bbox[3]) / 2
        if out:
            last = out[-1]
            lyc = sum((x.bbox[1] + x.bbox[3]) / 2 for x in last) / len(last)
            if abs(yc - lyc) <= 2.5:
                last.append(w)
                continue
        out.append([w])
    lines = [sorted(l, key=lambda w: w.bbox[0]) for l in out]
    mid = page.rect.width / 2

    def prose(ws):
        words = [w.text for w in ws]
        return sum(bool(re.fullmatch(r"[A-Za-z][a-z\-]{2,}[,.;:]?", t)) for t in words) >= 4 and \
            sum(bool(NUM.match(t)) for t in words) <= 1

    def gutter(ws):  # a wide gap across the page middle with running text on one side
        for k in range(1, len(ws)):
            if ws[k - 1].bbox[2] < mid + 15 and ws[k].bbox[0] > mid - 15 and ws[k].bbox[0] - ws[k - 1].bbox[2] > 9:
                return k if (prose(ws[:k]) or prose(ws[k:])) else None
        return None

    split = []
    for ws in lines:
        k = gutter(ws)
        split += [ws[:k], ws[k:]] if k else [ws]
    if len(split) > len(lines):  # two-column page: read the left column first
        split.sort(key=lambda ws: (ws[0].bbox[0] > mid, ws[0].bbox[1]))
    return [Line(ws) for ws in split if ws]


def _runs(words: list[Word]) -> list[tuple[float, str]]:
    """Tightly spaced words merged into labels, each with its horizontal centre."""
    out: list[list[Word]] = []
    for w in words:
        if out and w.bbox[0] - out[-1][-1].bbox[2] < 5:
            out[-1].append(w)
        else:
            out.append([w])
    return [((r[0].bbox[0] + r[-1].bbox[2]) / 2, " ".join(w.text for w in r)) for r in out]


def _gap(line: Line) -> float:
    """Median horizontal gap between words: tight for running text, wide for header cells."""
    ws = line.words
    gaps = sorted(b.bbox[0] - a.bbox[2] for a, b in zip(ws, ws[1:]))
    return gaps[len(gaps) // 2] if gaps else 0.0


def _is_data(l: Line) -> bool:
    nums = l.numbers()
    n = len(nums)
    if n < 2 or all(re.fullmatch(r"\d", x) for x in nums):  # "Stage 1 Stage 2 ..." is a header
        return False
    return (n >= 3 and n >= 0.3 * len(l.words)) or n >= 0.5 * len(l.words)


def _lead(line: Line) -> int:
    """Length of the leading run of tightly spaced words: the row label, numbers included."""
    if all(NUM.match(w.text) for w in line.words):
        return 0  # a row whose label sits on the line above
    k = 1
    while k < len(line.words) and line.words[k].bbox[0] - line.words[k - 1].bbox[2] < 5:
        k += 1
    return 0 if k == len(line.words) or not NUM.match(line.words[k].text) else k


def _is_prose(line: Line) -> bool:
    ws = line.words
    nums = sum(bool(NUM.match(w.text)) for w in ws)
    return len(ws) >= 12 and nums <= 1 and re.search(r"[a-z]{3,}[.,;]", line.text) is not None


def extract(path: str, doc_id: str | None = None, max_lines: int = 45) -> list[Table]:
    doc = pymupdf.open(path)
    doc_id = doc_id or path.rsplit("/", 1)[-1].removesuffix(".pdf")
    tables: list[Table] = []
    for pno, page in enumerate(doc, start=1):
        lines = _lines(page)
        for i, line in enumerate(lines):
            m = CAPTION.match(line.text)
            if not m:
                continue
            # caption may run over several lines; collect them
            j, cap = i + 1, [line.text]
            while j < len(lines) and _is_prose(lines[j]) and len(cap) < 6:
                cap.append(lines[j].text)
                j += 1
            below = _region(lines, j, +1, max_lines)
            above = _region(lines, i - 1, -1, max_lines)
            region = below if _score(below) >= _score(above) else above
            if _score(region) < 2:
                continue
            region = _adopt(lines, region)
            tables.append(Table(doc_id, pno, f"Table {m.group(2)}", " ".join(cap), region))
    return tables


def _region(lines: list[Line], start: int, step: int, max_lines: int) -> list[Line]:
    got = []
    k = start
    while 0 <= k < len(lines) and len(got) < max_lines:
        ln = lines[k]
        if CAPTION.match(ln.text) or _is_prose(ln):
            break
        got.append(ln)
        k += step
    return got if step > 0 else got[::-1]


def _adopt(lines: list[Line], region: list[Line]) -> list[Line]:
    """Put back a header line that sits above the table's first data row, inside the data
    rows' width, but was read with the other text column, such as a group header
    ("LLaMA-2  Mistral  Gemma") that starts right of the page middle."""
    data = [l for l in region if _is_data(l)]
    if not data:
        return region
    x0, x1 = min(l.bbox[0] for l in data), max(l.bbox[2] for l in data)
    y0, ytop = min(l.bbox[1] for l in region), data[0].bbox[1]
    have = {id(l) for l in region}
    extra = [l for l in lines if id(l) not in have and y0 < (l.bbox[1] + l.bbox[3]) / 2 < ytop
             and l.bbox[0] >= x0 - 10 and l.bbox[2] <= x1 + 10 and 2 <= len(l.words) <= 12
             and sum(bool(re.fullmatch(r"[a-z][a-z\-]+[,.;:]?", w.text)) for w in l.words) <= 1
             and not CAPTION.match(l.text)]
    if not extra:
        return region
    return sorted(region + extra, key=lambda l: l.bbox[1])


def _score(region: list[Line]) -> int:
    return sum(len(l.numbers()) >= 2 for l in region)
