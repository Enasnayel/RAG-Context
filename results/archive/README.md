# Archive

Superseded result files, kept for provenance. **None of these feed the
current analysis** — `RAG.ipynb` reads only `results/results.csv`,
`results/results_with_composite.csv`, and `results/cost_ledger.csv`, none of
which live here.

## `results.csv.bak-prendcgfix`

76 rows, from before the nDCG@5 computation was corrected. `ndcg_at_5`
values in this file can exceed 1.0 (e.g. `financebench_json_fixed_256_claude`
reads 1.5367 here, against 0.9234 for the same run in the corrected
`results.csv`) — nDCG is bounded [0, 1] by definition, so any value above 1
is the bug, not a result. This was the multi-chunks-per-document
double-counting issue `ndcg_at_k()`'s docstring in the notebook describes.

**Must not be used.** `recall_at_5` and `mrr` in this file are unaffected by
the bug, but there is no reason to read stale values for those either when
`results.csv` has the corrected, complete data.

## `results.csv.bak`

An earlier backup with an inconsistent schema: some rows have 20 columns,
others 32 (`pandas.read_csv` fails on it directly - `Error tokenizing data.
... Expected 20 fields in line 8, saw 32`). This is the header/schema
mismatch the driver cell's `write_rows()` comment warns about - a header is
only ever written once, so when `RESULT_FIELDS` grew from 20 to 32 columns,
old- and new-schema rows ended up in the same file at different widths.
Kept as a record of that incident, not as usable data.

## `json_fixed_256.json`, `toon_fixed_256.json`, `yaml_fixed_256.json`

Individual per-cell result files from the pilot phase (HotpotQA only, fixed
chunking, 256 tokens - three formats, one cell each), predating the full
4-corpus factorial rebuild. Superseded by the `hotpotqa` rows in
`results.csv`, which cover the same combination as part of the full 144-row
core factorial. Originally at `results/hotpotqa/`, moved here since that
directory no longer serves a purpose once these are archived.
