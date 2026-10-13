# rowcheck

Check a numeric claim against the tables of a paper, and get back the exact row and cell
it rests on, marked on the page.

```
$ python -m rowcheck mistral-7b.pdf "Mistral 7B reaches 52.2% on MATH."
CONTRADICTED  Mistral 7B reaches 52.2% on MATH.
              Table 2 (p. 4) reports 13.1%, not 52.2%, for Mistral 7B / MATH.
              The same row prints 52.2% under GSM8K.
```

With `--out report.html` each verdict comes with a crop of the page: the row is boxed, the
cited cell is filled, and the number the claim actually used is outlined.

## How it works

1. **Tables as positioned words.** Paper tables are mostly borderless, with stacked headers
   and two-column page layouts. rowcheck groups words into visual lines, finds each table
   from its caption, locates columns from where the numbers line up across rows, and hangs
   the header words over each column. A row then reads `Mistral 7B Pretrained | MMLU=60.1% |
   HellaSwag=81.3% | ...`. When a row's labels are not one-to-one, it is passed on raw
   rather than mislabelled.
2. **An open-weight model reads the table.** Gemma 4 (26B-A4B, run locally through
   llama.cpp) gets the claim and the best-matching tables and answers, under a JSON schema,
   which table, line, column and cell the claim is about, and whether each setting the claim
   states (shots, fine-tuned or base, split) is confirmed, contradicted or not stated.
3. **Code decides the numbers.** The cited cell must be printed on that exact line, the
   claimed value is compared with it, units (B, T, million) are checked against the header,
   and the claimed value is searched across the table so a number lifted from the wrong row
   or column is named.

Verdicts: `supported`, `contradicted`, `ambiguous` (the number is there, a stated setting is
not), `unsupported` (no row measures it).

## Run it

```
llama-server -m gemma-4-26B-A4B-it-Q4_0.gguf --port 8471 --jinja -ngl 99 --n-cpu-moe 6 -c 16384
pip install -r requirements.txt && ./fetch-papers.sh
python -m rowcheck paper.pdf "claim one" "claim two" --out report.html
```

`ROWCHECK_LLM` points at any OpenAI-compatible endpoint that supports JSON-schema output.

## Evaluation

`eval/claims.jsonl` holds 30 claims over six CC BY 4.0 papers (Mistral 7B, OLMo, OLMo 2,
Phi-3, Gemma 2, SmolLM2), each labelled from the page before any model run, plus six trap
claims that swap a row, unit, denominator or setting. The papers were split before
inference (`eval/split.json`): four for development, Phi-3 and OLMo 2 held out.

The control is plain chunk retrieval with the same model: 1,200-character page chunks, the
three best by term overlap, evidence written before the verdict.

| | rowcheck | chunk retrieval |
|---|---:|---:|
| Development claims (20) | 19 | 18 |
| Held-out claims (10) | 7 | 6 |
| Trap claims called "supported" (6) | 0 | 1 |

A second check used four unseen CC BY papers (Gemma, Mixtral, Nemotron-4, Qwen): 22 claims in
`eval/claims-unseen.jsonl`, labelled before inference, nine of them traps.

| Unseen papers (13 claims, 9 traps) | rowcheck | chunk retrieval |
|---|---:|---:|
| First run, claims right | 9 | 12 |
| First run, traps called "supported" | 4 | 4 |
| After table-reader fixes, claims right | 13 | 12 |
| After table-reader fixes, traps called "supported" | 0 | 4 |

The first run is kept in `eval/results-unseen-prefix.jsonl`. The fixes (group headers, family
labels printed once beside size rows, U+2217 asterisks, the cited cell checked against the
claim's column, shot counts looked up in other tables) were made against these papers, so the
second pair of rows is development data. With the same code, development claims are 20/20 and
the original held-out claims 8/10.

`python -m eval.run dev|confirmation|unseen` reruns both; `python -m eval.rebaseline` reruns the
control alone; `python -m eval.demo out/index.html` builds the demo page. A shot-count check
was added after the held-out run and is not reflected in its numbers.

## Licence and credits

Code: MIT. The example papers are CC BY 4.0 and are downloaded from arXiv, not redistributed.
Built with Claude Code as a coding assistant.
