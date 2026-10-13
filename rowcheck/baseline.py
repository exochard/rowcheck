"""Ordinary chunk RAG with the same model: page text in 1,200-character chunks, the three
best by term overlap, and the model judges the claim from them."""

from __future__ import annotations

import json
import math
import urllib.request
from collections import Counter

import pymupdf

from .check import ENDPOINT, _terms

SCHEMA = {"type": "object", "properties": {
    "evidence": {"type": "string"}, "page": {"type": "integer"},
    "verdict": {"type": "string", "enum": ["supported", "contradicted", "ambiguous", "unsupported"]}},
    "required": ["evidence", "page", "verdict"]}  # evidence first: the model reasons before it commits

PROMPT = """Decide whether the paper supports this numeric claim.
Claim: {claim}

Excerpts:
{chunks}

supported: the paper reports this number for this subject, metric and setting.
contradicted: the paper reports a different number or a different setting.
ambiguous: the number is there but a setting in the claim cannot be confirmed.
unsupported: the excerpts do not measure this. Answer in JSON."""


def chunks(pdf: str, size: int = 1200) -> list[tuple[int, str]]:
    out = []
    for pno, page in enumerate(pymupdf.open(pdf), start=1):
        text = " ".join(page.get_text().split())
        out += [(pno, text[i:i + size]) for i in range(0, len(text), size)]
    return out


def check(claim: str, pdf: str, k: int = 3) -> dict:
    cs = chunks(pdf)
    q = Counter(_terms(claim))
    df = Counter(w for _, c in cs for w in set(_terms(c)))
    n = len(cs)
    score = [sum(math.log(1 + n / df[w]) for w in q if w in set(_terms(c))) for _, c in cs]
    top = sorted(range(n), key=lambda i: -score[i])[:k]
    blocks = "\n\n".join(f"[page {cs[i][0]}] {cs[i][1]}" for i in top)
    body = {"messages": [{"role": "user", "content": PROMPT.format(claim=claim, chunks=blocks)}],
            "temperature": 0, "max_tokens": 1500, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "judge", "schema": SCHEMA}}}
    req = urllib.request.Request(ENDPOINT, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.load(r)["choices"][0]["message"]["content"] or ""
    return json.loads(text[text.find("{"): text.rfind("}") + 1])
