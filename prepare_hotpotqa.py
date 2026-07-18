"""
prepare_hotpotqa.py
-------------------
Builds the pilot dataset: ~500 HotpotQA documents + 50 questions.

Design logic:
- We sample 50 questions first (seed=42 so the sample is reproducible).
- Every question's GOLD paragraphs must be in the corpus, otherwise
  Recall@5 / MRR are meaningless (the right answer would be unfindable).
- Each question contributes its full 10-paragraph context (2 gold +
  8 distractor) to the corpus, so 50 questions -> up to 500 unique
  documents after deduplication across overlapping paragraphs.

Outputs (written to ./data/):
- corpus.jsonl     one line per document: {doc_id, title, text}
- questions.jsonl  one line per question: {qid, question, answer, gold_titles}

Run:  python prepare_hotpotqa.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset  # pip install datasets

SEED = 42
N_QUESTIONS = 50
N_DOCS = 500
OUT_DIR = Path("data")


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    # The distractor setting gives each question 10 paragraphs:
    # 2 gold (supporting) + 8 distractors. That is exactly the structure
    # we need to build a corpus where gold and non-gold coexist.
    print("Loading HotpotQA (distractor, validation split)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    # Reproducible sample of question indices.
    indices = random.sample(range(len(ds)), N_QUESTIONS * 2)  # oversample, filter below

    questions = []
    corpus = {}          # title -> full paragraph text (titles are unique in HotpotQA)
    gold_titles_all = set()

    for idx in indices:
        if len(questions) >= N_QUESTIONS:
            break
        ex = ds[idx]

        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        sup_titles = set(ex["supporting_facts"]["title"])

        # Skip malformed examples (defensive: gold must appear in context).
        if not sup_titles.issubset(set(titles)):
            continue

        # Register the question with its gold document titles.
        questions.append({
            "qid": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "gold_titles": sorted(sup_titles),
        })
        gold_titles_all.update(sup_titles)

        # Stage every paragraph of this question as a corpus candidate.
        for title, sents in zip(titles, sentences):
            if title not in corpus:
                corpus[title] = " ".join(sents).strip()

    # --- Assemble the corpus --------------------------------------------
    # Priority 1: all gold documents (required for valid IR metrics).
    # Priority 2: distractors, shuffled deterministically, until N_DOCS.
    gold_docs = sorted(gold_titles_all)
    distractors = sorted(t for t in corpus if t not in gold_titles_all)
    random.shuffle(distractors)

    selected = gold_docs + distractors[: max(0, N_DOCS - len(gold_docs))]
    selected = selected[:N_DOCS]

    # Sanity check: every gold title must be in the final corpus.
    missing = gold_titles_all - set(selected)
    if missing:
        raise RuntimeError(f"Gold documents missing from corpus: {missing}")

    # --- Write outputs -------------------------------------------------
    with open(OUT_DIR / "corpus.jsonl", "w", encoding="utf-8") as f:
        for i, title in enumerate(selected):
            f.write(json.dumps({
                "doc_id": f"doc_{i:03d}",
                "title": title,
                "text": corpus[title],
            }, ensure_ascii=False) + "\n")

    with open(OUT_DIR / "questions.jsonl", "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"Corpus: {len(selected)} documents "
          f"({len(gold_docs)} gold, {len(selected) - len(gold_docs)} distractor)")
    print(f"Questions: {len(questions)}")
    print(f"Written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()