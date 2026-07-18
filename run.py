"""
run.py — THE single entry point for every experiment in the thesis.

    python run.py --scope pilot            # 2 cells (baseline + TOON contrast)
    python run.py --scope core             # full 18-cell factorial
    python run.py --cells json_fixed_256   # any explicit cell list

Same code path at every scale. Properties:
- RESUMABLE: a cell whose results file exists is skipped. Delete the
  file to force a rerun. A crash costs one cell, never a run.
- CACHED: contextualization snippets and embeddings are disk-cached;
  reruns never re-pay API cost or embedding compute.
- Pipeline order (Option A): chunk -> contextualize (raw text) ->
  serialize (snippet inside record) -> embed full record -> index ->
  retrieve -> generate -> evaluate.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from config import (SCOPES, TOP_K, DATA_DIR, RESULTS_DIR, cell_id, BASELINE)
from chunking import chunk
from serialise import serialize, roundtrip_ok
from tokens import count_gpt
from contextualise import contextualize
from embed_index import embed_corpus, build_index, embed_query
from generate import generate
from evaluate import recall_and_mrr, faithfulness


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "UNCOMMITTED"


def run_cell(fmt, strategy, size, docs, questions, corpus):
    cid = cell_id(fmt, strategy, size)
    out_path = Path(RESULTS_DIR) / corpus / f"{cid}.json"
    if out_path.exists():
        print(f"[skip] {corpus}/{cid} — results exist")
        return json.loads(out_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {corpus} / {cid} ===")

    # -- chunk (F2 x chunk size) ---------------------------------------
    chunks_by_doc = {d["doc_id"]: chunk(d["text"], strategy, size) for d in docs}
    chunk_tok = sum(count_gpt(c) for cs in chunks_by_doc.values() for c in cs)

    # -- contextualize on RAW text (shared across formats via cache) ----
    snippets, cached = contextualize(docs, chunks_by_doc, corpus, strategy, size)
    print(f"  snippets: {'cache hit' if cached else 'generated'} "
          f"({sum(len(v) for v in chunks_by_doc.values())} chunks)")
    snippet_tok = sum(count_gpt(s) for s in snippets.values())

    # -- build records, validate, serialize (F3) ------------------------
    records = []
    for d in docs:
        for ci, text in enumerate(chunks_by_doc[d["doc_id"]]):
            records.append({
                "doc_id": d["doc_id"], "title": d["title"],
                "chunk_id": f"{d['doc_id']}_c{ci:02d}",
                "context": snippets[(d["doc_id"], str(ci))],
                "text": text,
            })
    assert all(roundtrip_ok(r) for r in records), "round-trip validation failed"
    serialized = [serialize(r, fmt) for r in records]

    # -- embed the FULL serialized record + exact index ------------------
    vecs = embed_corpus(serialized, cache_key=f"{corpus}_{cid}")
    index = build_index(vecs)

    # -- query loop -------------------------------------------------------
    per_q = []
    for q in questions:
        t0 = time.time()
        _, idxs = index.search(embed_query(q["question"]), TOP_K)
        top = idxs[0]
        rec, rr = recall_and_mrr([records[i]["title"] for i in top],
                                 q["gold_titles"])
        context = "\n\n".join(serialized[i] for i in top)
        ans = {p: generate(p, context, q["question"]) for p in ("gpt", "claude")}
        latency = time.time() - t0
        per_q.append({
            "qid": q["qid"], "recall@5": rec, "mrr": rr,
            "ctx_tokens": count_gpt(context), "latency_s": latency,
            "faith_gpt": faithfulness(context, ans["gpt"]),
            "faith_claude": faithfulness(context, ans["claude"]),
            "answer_gpt": ans["gpt"], "answer_claude": ans["claude"],
        })
        print(f"  {q['qid'][:8]}  R@5={rec:.2f} MRR={rr:.2f} "
              f"ctx={per_q[-1]['ctx_tokens']}t {latency:.1f}s")

    mean = lambda k: float(np.mean([x[k] for x in per_q]))
    result = {
        "corpus": corpus, "cell": cid,
        "format": fmt, "strategy": strategy, "size": size,
        "commit": git_hash(),
        "n_docs": len(docs), "n_chunks": len(records),
        "n_questions": len(questions),
        "indexing_tokens": int(sum(count_gpt(s) for s in serialized)),
        "ctx_overhead_pct": round(100 * snippet_tok / chunk_tok, 1),
        "recall@5": round(mean("recall@5"), 4),
        "mrr": round(mean("mrr"), 4),
        "mean_ctx_tokens": round(mean("ctx_tokens"), 1),
        "faithfulness_gpt": round(mean("faith_gpt"), 4),
        "faithfulness_claude": round(mean("faith_claude"), 4),
        "mean_latency_s": round(mean("latency_s"), 2),
        "per_question": per_q,
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return result


def section4_block(results, corpus):
    b = next(r for r in results
             if (r["format"], r["strategy"], r["size"]) == BASELINE)
    lines = [
        "================ MID-REVIEW SECTION - RESULT BLOCK ==================",
        f"The baseline cell (JSON x fixed-size, {b['size']} tokens) runs "
        f"end-to-end on a {corpus} subset ({b['n_docs']} documents, "
        f"{b['n_questions']} questions, {b['n_chunks']} chunks). "
        f"Commit {b['commit']}.",
        "",
        f"Baseline: Recall@5={b['recall@5']}  MRR={b['mrr']}  "
        f"faith(GPT)={b['faithfulness_gpt']}  faith(Claude)={b['faithfulness_claude']}",
        f"          ctx tokens/query={b['mean_ctx_tokens']}  "
        f"contextualization overhead={b['ctx_overhead_pct']}%  "
        f"indexing tokens={b['indexing_tokens']}",
    ]
    for r in results:
        if r is b:
            continue
        saving = round(100 * (1 - r["indexing_tokens"] / b["indexing_tokens"]), 1)
        lines += [
            "",
            f"{r['cell']}: Recall@5={r['recall@5']}  MRR={r['mrr']}  "
            f"faith(GPT)={r['faithfulness_gpt']}  faith(Claude)={r['faithfulness_claude']}",
            f"          ctx tokens/query={r['mean_ctx_tokens']}  "
            f"indexing tokens={r['indexing_tokens']}  "
            f"token saving vs baseline={saving}%",
        ]
    lines.append("==================================================================")
    return "\n".join(lines)


def inspect_walkthrough():
    """See the pipeline with your own eyes: 2 docs + 1 question through
    every stage, printing the REAL artifacts. Uses a separate cache
    namespace ('inspect') so it never pollutes pilot/factorial caches.
    Cost: a few cents. Time: ~2-3 minutes."""
    from config import GEN_PARAMS
    docs = load_jsonl(Path(DATA_DIR) / "corpus.jsonl")[:2]
    questions = load_jsonl(Path(DATA_DIR) / "questions.jsonl")

    # pick a question whose gold docs are among our 2 docs if possible
    titles = {d["title"] for d in docs}
    q = next((x for x in questions if set(x["gold_titles"]) & titles),
             questions[0])

    bar = lambda t: print("\n" + "=" * 70 + f"\n STAGE: {t}\n" + "=" * 70)

    bar("1. RAW DOCUMENT (input)")
    d = docs[0]
    print(f"doc_id={d['doc_id']}  title={d['title']}")
    print(d["text"][:400] + ("..." if len(d["text"]) > 400 else ""))

    bar("2. CHUNKING (fixed, 256 tokens)")
    chunks_by_doc = {x["doc_id"]: chunk(x["text"], "fixed", 256) for x in docs}
    c0 = chunks_by_doc[d["doc_id"]][0]
    print(f"{d['title']} -> {len(chunks_by_doc[d['doc_id']])} chunk(s); "
          f"chunk 0 = {count_gpt(c0)} tokens:")
    print(c0[:400] + ("..." if len(c0) > 400 else ""))

    bar("3. CONTEXTUALIZATION (Claude writes the snippet — live API call)")
    snippets, cached = contextualize(docs, chunks_by_doc, "inspect", "fixed", 256)
    snip = snippets[(d["doc_id"], "0")]
    print(f"({'from cache' if cached else 'freshly generated'})")
    print(f"Snippet for chunk 0:\n  \"{snip}\"")
    print(f"Snippet cost: +{count_gpt(snip)} tokens on a "
          f"{count_gpt(c0)}-token chunk — this is the overhead the "
          f"formats then compress differently.")

    bar("4. SERIALIZATION — the SAME record in all three formats")
    rec = {"doc_id": d["doc_id"], "title": d["title"],
           "chunk_id": f"{d['doc_id']}_c00", "context": snip, "text": c0}
    for fmt in ("json", "yaml", "toon"):
        s = serialize(rec, fmt)
        print(f"\n--- {fmt.upper()}  ({count_gpt(s)} tokens) ---")
        print(s[:350] + ("..." if len(s) > 350 else ""))
    print("\n^ identical information, three token prices. "
          "That difference, at index scale, is the efficiency side "
          "of the thesis.")

    bar("5. EMBEDDING + INDEX (jina-v2 -> vectors -> FAISS)")
    records, serialized = [], []
    for x in docs:
        for ci, t in enumerate(chunks_by_doc[x["doc_id"]]):
            r = {"doc_id": x["doc_id"], "title": x["title"],
                 "chunk_id": f"{x['doc_id']}_c{ci:02d}",
                 "context": snippets[(x["doc_id"], str(ci))], "text": t}
            records.append(r)
            serialized.append(serialize(r, "json"))
    vecs = embed_corpus(serialized, cache_key="inspect_json_fixed_256")
    index = build_index(vecs)
    print(f"{len(records)} serialized records -> matrix {vecs.shape} "
          f"(one {vecs.shape[1]}-number vector per record)")
    print(f"Record 0 vector, first 6 of {vecs.shape[1]} numbers: "
          f"{[round(float(v), 3) for v in vecs[0][:6]]}")

    bar("6. RETRIEVAL (question -> nearest vectors)")
    print(f"Question: {q['question']}")
    print(f"Gold documents (ground truth): {q['gold_titles']}")
    k = min(3, len(records))
    scores, idxs = index.search(embed_query(q["question"]), k)
    for rank, (i, sc) in enumerate(zip(idxs[0], scores[0]), 1):
        hit = "GOLD" if records[i]["title"] in q["gold_titles"] else "    "
        print(f"  #{rank} [{hit}] sim={sc:.3f}  {records[i]['chunk_id']}  "
              f"({records[i]['title']})")
    rec_at, rr = recall_and_mrr([records[i]["title"] for i in idxs[0]],
                                q["gold_titles"])
    print(f"-> Recall@{k}={rec_at:.2f}  MRR={rr:.2f}  "
          f"(only meaningful when the gold docs are in the corpus — "
          f"with 2 docs this is a demo, not a metric)")

    bar("7. GENERATION (retrieved records pasted into the prompt — live)")
    context = "\n\n".join(serialized[i] for i in idxs[0])
    print(f"Context sent to the model: {count_gpt(context)} tokens, "
          f"pinned params {GEN_PARAMS}")
    ans = generate("gpt", context, q["question"])
    print(f"GPT answer: {ans}")

    bar("8. FAITHFULNESS JUDGE (claims checked against context — live)")
    f = faithfulness(context, ans)
    print(f"Judge decomposed the answer into claims and verified each "
          f"against the context only.\nFaithfulness = {f:.2f} "
          f"(fraction of claims the context actually supports)")

    print("\n" + "=" * 70)
    print(" That is one record and one question, end to end.")
    print(" 'python run.py --scope pilot' = the same path x ~700 records")
    print(" x 25 questions x 2 formats, with every number logged.")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=list(SCOPES), default=None)
    ap.add_argument("--cells", default=None,
                    help="comma-separated cell ids, e.g. json_fixed_256,toon_fixed_256")
    ap.add_argument("--corpus", default="hotpotqa")
    ap.add_argument("--inspect", action="store_true",
                    help="walk 2 docs + 1 question through every stage, "
                         "printing the real artifacts (~3 min, a few cents)")
    args = ap.parse_args()

    if args.inspect:
        inspect_walkthrough()
        return

    if args.cells:
        cells = []
        for cid in args.cells.split(","):
            fmt, strategy, size = cid.rsplit("_", 2)[0], *cid.rsplit("_", 2)[1:]
            cells.append((fmt, strategy, int(size)))
    else:
        cells = SCOPES[args.scope or "pilot"]

    docs = load_jsonl(Path(DATA_DIR) / "corpus.jsonl")
    questions = load_jsonl(Path(DATA_DIR) / "questions.jsonl")

    results = [run_cell(f, s, z, docs, questions, args.corpus)
               for (f, s, z) in cells]

    block = section4_block(results, args.corpus)
    print("\n" + block)
    out = Path(RESULTS_DIR) / args.corpus / "summary_block.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(block, encoding="utf-8")


if __name__ == "__main__":
    main()
