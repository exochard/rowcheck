"""Run rowcheck and the chunk-RAG baseline on the frozen claims; write eval/results.jsonl
and the gate JSON. Usage: python -m eval.run [dev|confirmation|unseen|all]"""

import json
import random
import sys
import time
from pathlib import Path

from rowcheck import baseline
from rowcheck.check import check
from rowcheck.tables import extract

HERE = Path(__file__).parent
PAPERS = HERE.parent / "data" / "papers"


def main(which="all"):
    split = json.loads((HERE / "split.json").read_text())
    src = "claims-unseen.jsonl" if which == "unseen" else "claims.jsonl"
    claims = [json.loads(l) for l in (HERE / src).read_text().splitlines()]
    docs = split.get(which, split["docs"]) if which != "dev" else split["development"]
    claims = [c for c in claims if c["doc"] in docs]
    tables = {d: extract(str(PAPERS / f"{d}.pdf"), d) for d in docs}
    out = HERE / f"results-{which}.jsonl"
    rows = []
    for c in claims:
        t0 = time.time()
        v = check(c["claim"], tables[c["doc"]])
        t1 = time.time()
        b = baseline.check(c["claim"], str(PAPERS / f"{c['doc']}.pdf"))
        t2 = time.time()
        row = dict(c, rowcheck=v.verdict, rowcheck_table=v.table, rowcheck_page=v.page, rowcheck_line=v.line,
                   rowcheck_summary=v.summary, rowcheck_s=round(t1 - t0, 1),
                   baseline=b.get("verdict"), baseline_page=b.get("page"), baseline_s=round(t2 - t1, 1))
        rows.append(row)
        print(f"{c['id']} gold={c['gold']:12} rowcheck={v.verdict:12} base={b.get('verdict'):12} | {c['claim'][:60]}", flush=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    gate(rows, which, split)


def gate(rows, which, split):
    def acc(key, rs):
        return sum(r[key] == r["gold"] for r in rs)
    claims = [r for r in rows if r["kind"] == "claim"]
    neg = [r for r in rows if r["kind"] == "negative"]
    rc, bc = acc("rowcheck", claims), acc("baseline", claims)
    net = sum((r["rowcheck"] == r["gold"]) and (r["baseline"] != r["gold"]) for r in claims) - \
        sum((r["rowcheck"] != r["gold"]) and (r["baseline"] == r["gold"]) for r in claims)
    # descriptive document-level bootstrap of rowcheck accuracy
    by_doc = {}
    for r in claims:
        by_doc.setdefault(r["doc"], []).append(r["rowcheck"] == r["gold"])
    rnd, docs, boots = random.Random(20261012), list(by_doc), []
    for _ in range(5000):
        pick = [by_doc[rnd.choice(docs)] for _ in docs]
        flat = [x for p in pick for x in p]
        boots.append(sum(flat) / len(flat))
    boots.sort()
    right_place = [r for r in claims if r["rowcheck"] == r["gold"] and r["rowcheck"] in ("supported", "contradicted")]
    g = {
        "split": which, "docs": sorted(by_doc), "seed": split["seed"], "bootstrap_seed": 20261012,
        "claims": len(claims), "rowcheck_correct": rc, "baseline_correct": bc,
        "rowcheck_accuracy": round(rc / max(1, len(claims)), 3),
        "baseline_accuracy": round(bc / max(1, len(claims)), 3),
        "net_improvement_vs_baseline": net,
        "rowcheck_accuracy_doc_bootstrap_95": [round(boots[124], 3), round(boots[4874], 3)],
        "citation_table_matches_gold": sum(r["rowcheck_table"] == r["table"] for r in right_place),
        "citation_checked": len(right_place),
        "negatives": len(neg), "negatives_false_supported": sum(r["rowcheck"] == "supported" for r in neg),
        "baseline_negatives_false_supported": sum(r["baseline"] == "supported" for r in neg),
        "rowcheck_mean_s": round(sum(r["rowcheck_s"] for r in rows) / max(1, len(rows)), 1),
        "baseline_mean_s": round(sum(r["baseline_s"] for r in rows) / max(1, len(rows)), 1),
        "failures": [{k: r[k] for k in ("id", "claim", "gold", "rowcheck", "rowcheck_summary")}
                     for r in rows if r["rowcheck"] != r["gold"]],
        "thresholds": {"accuracy": 0.85, "net_improvement": 3, "negatives_false_supported": 0},
    }
    g["verdict"] = "pass" if (g["rowcheck_accuracy"] >= 0.85 and net >= 3 and g["negatives_false_supported"] == 0
                              and g["citation_table_matches_gold"] == g["citation_checked"]) else "fail"
    (HERE / f"gate-{which}.json").write_text(json.dumps(g, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in g.items() if k != "failures"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
