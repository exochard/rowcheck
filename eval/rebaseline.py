"""Rerun only the chunk-RAG control (evidence-first schema) on every frozen claim."""
import json
from pathlib import Path

from rowcheck import baseline

HERE = Path(__file__).parent
rows = []
for split in ("dev", "confirmation"):
    for line in (HERE / f"results-{split}.jsonl").read_text().splitlines():
        r = json.loads(line)
        b = baseline.check(r["claim"], str(HERE.parent / "data" / "papers" / f"{r['doc']}.pdf"))
        r["baseline_v1"], r["baseline"] = r["baseline"], b.get("verdict")
        r["split"] = split
        rows.append(r)
        print(r["id"], r["gold"], "rowcheck", r["rowcheck"], "base_v1", r["baseline_v1"], "base_v2", r["baseline"], flush=True)
(HERE / "results-rebaseline.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
for split in ("dev", "confirmation"):
    rs = [r for r in rows if r["split"] == split and r["kind"] == "claim"]
    neg = [r for r in rows if r["split"] == split and r["kind"] == "negative"]
    print(split, "rowcheck", sum(r["rowcheck"] == r["gold"] for r in rs), "baseline_v2", sum(r["baseline"] == r["gold"] for r in rs),
          "of", len(rs), "| negatives supported: rowcheck", sum(r["rowcheck"] == "supported" for r in neg),
          "baseline_v2", sum(r["baseline"] == "supported" for r in neg))
