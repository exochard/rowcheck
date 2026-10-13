"""rowcheck PAPER.pdf "claim" ["claim" ...] [--out report.html] [--json]"""

import argparse
import json
import sys

from .check import check
from .report import crop_png, html_report
from .tables import extract


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rowcheck", description="Check numeric claims against a paper's tables.")
    ap.add_argument("pdf")
    ap.add_argument("claims", nargs="*", help="claims; read one per line from stdin when omitted")
    ap.add_argument("--out", help="write an HTML report with marked page crops")
    ap.add_argument("--json", action="store_true", help="print one JSON object per claim")
    a = ap.parse_args(argv)
    claims = a.claims or [l.strip() for l in sys.stdin if l.strip()]
    tables = extract(a.pdf)
    items = []
    for c in claims:
        v = check(c, tables)
        t = next((t for t in tables if t.label == v.table and t.page == v.page), None)
        png = crop_png(a.pdf, t, v) if (a.out and t) else None
        items.append((v, png))
        if a.json:
            print(json.dumps(v.to_json(), ensure_ascii=False))
        else:
            print(f"{v.verdict.upper():13} {c}\n{'':13} {v.summary}")
    if a.out:
        open(a.out, "w").write(html_report(items, f"rowcheck: {a.pdf.rsplit('/', 1)[-1]}"))
        print(f"report: {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
