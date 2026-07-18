"""
evaluate.py — metrics.

recall_and_mrr: IR metrics against gold document titles.
faithfulness:   RAGAS-style claim-decomposition judge, single call,
                pinned model. NOTE: swap in the full ragas library for
                the factorial write-up; this variant is documented as
                the pilot-stage judge in the methodology footnote.
"""

import json

from config import JUDGE_MODEL, SEED

_oai = None

def oai():
    global _oai
    if _oai is None:
        from openai import OpenAI  # lazy
        _oai = OpenAI()
    return _oai


def recall_and_mrr(retrieved_titles, gold_titles, k=None):
    gold = set(gold_titles)
    if k:
        retrieved_titles = retrieved_titles[:k]
    recall = len(gold & set(retrieved_titles)) / len(gold)
    rr = 0.0
    for rank, t in enumerate(retrieved_titles, start=1):
        if t in gold:
            rr = 1.0 / rank
            break
    return recall, rr


JUDGE_PROMPT = (
    "You are grading faithfulness. Decompose the ANSWER into atomic factual "
    "claims, then check each claim against the CONTEXT only.\n"
    'Return strict JSON: {{"claims": <int>, "supported": <int>}}.\n\n'
    "CONTEXT:\n{context}\n\nANSWER:\n{answer}"
)


def faithfulness(context: str, answer: str) -> float:
    r = oai().chat.completions.create(
        model=JUDGE_MODEL, seed=SEED, temperature=0,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(context=context, answer=answer)}],
        response_format={"type": "json_object"},
    )
    j = json.loads(r.choices[0].message.content)
    return (j["supported"] / j["claims"]) if j.get("claims") else 1.0
