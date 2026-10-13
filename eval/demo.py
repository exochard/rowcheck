"""Build the demo page: python -m eval.demo out/index.html"""
import sys
from pathlib import Path

from rowcheck.check import check
from rowcheck.report import crop_png, html_report
from rowcheck.tables import extract

PAPERS = Path(__file__).parent.parent / "data" / "papers"
EXAMPLES = [
    ("2402.00838", "TÜLU 2 reaches an 85.1% win rate on AlpacaEval."),
    ("2310.06825", "Mistral 7B reaches 52.2% on MATH."),
    ("2408.00118", "Gemma-2 9B scores 68.6 on GSM8K in a 0-shot setting."),
    ("2502.02737", "Stack-Edu has 21.8T Python tokens."),
    ("2310.06825", "Mistral 7B scores 60.1% on MMLU in a 0-shot setting."),
    ("2408.00118", "Gemma-2 27B scores 75.2 on 5-shot MMLU."),
]


def main(out="out/index.html"):
    cache, items = {}, []
    for doc, claim in EXAMPLES:
        pdf = str(PAPERS / f"{doc}.pdf")
        tables = cache.setdefault(doc, extract(pdf, doc))
        v = check(claim, tables)
        t = next((t for t in tables if t.label == v.table and t.page == v.page), None)
        items.append((v, crop_png(pdf, t, v) if t else None))
        print(v.verdict, "|", claim, "|", v.summary)
    Path(out).write_text(html_report(items, "rowcheck: numeric claims checked against the paper's tables"))


if __name__ == "__main__":
    main(*sys.argv[1:])
