"""
generate.py — the two generators (held factor levels, pinned params).
"""

from config import GPT_MODEL, CLAUDE_MODEL, GEN_PARAMS, SEED

_oai, _ant = None, None

def oai():
    global _oai
    if _oai is None:
        from openai import OpenAI  # lazy
        _oai = OpenAI()
    return _oai

def ant():
    global _ant
    if _ant is None:
        from anthropic import Anthropic  # lazy
        _ant = Anthropic()
    return _ant


ANSWER_PROMPT = (
    "Answer the question using ONLY the retrieved context below. "
    "Be concise.\n\n<context>\n{context}\n</context>\n\nQuestion: {q}\nAnswer:"
)


def generate(provider: str, context: str, question: str) -> str:
    prompt = ANSWER_PROMPT.format(context=context, q=question)

    if provider == "gpt":
        r = oai().chat.completions.create(
            model=GPT_MODEL,
            seed=SEED,
            **GEN_PARAMS,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content.strip()

    if provider == "claude":
        r = ant().messages.create(
            model=CLAUDE_MODEL,
            max_tokens=GEN_PARAMS["max_tokens"],
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text.strip()

    raise ValueError(f"unknown provider: {provider}")
