"""
chunking.py — the chunking-strategy factor (F2).
One dispatch function; strategies are added here and nowhere else.

fixed      — cut every N tokens (baseline strategy).
recursive  — split on a separator priority (paragraph > sentence),
             packing pieces up to N tokens without cutting mid-sentence
             unless a single sentence alone exceeds N.
semantic   — boundary where adjacent-sentence embedding similarity drops.
             NOT implemented yet: it needs the embedder in the loop and
             a validated threshold. Gated with a clear error so no cell
             can silently run a wrong implementation.
"""

import re

from tokens import get_encoder, count_gpt


def chunk(text: str, strategy: str, size_tokens: int) -> list:
    if strategy == "fixed":
        return fixed_chunks(text, size_tokens)
    if strategy == "recursive":
        return recursive_chunks(text, size_tokens)
    if strategy == "semantic":
        raise NotImplementedError(
            "semantic chunking is not implemented yet — implement and "
            "validate before running semantic cells (factorial stage)."
        )
    raise ValueError(f"unknown strategy: {strategy}")


def fixed_chunks(text: str, size_tokens: int) -> list:
    """Cut every size_tokens tokens, regardless of structure."""
    enc = get_encoder()
    ids = enc.encode(text)
    out = []
    for start in range(0, len(ids), size_tokens):
        piece = enc.decode(ids[start:start + size_tokens]).strip()
        if piece:
            out.append(piece)
    return out


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def recursive_chunks(text: str, size_tokens: int) -> list:
    """Separator-priority splitting: try paragraphs first; any paragraph
    over budget is split into sentences; sentences are packed greedily
    up to the token budget. A lone sentence longer than the budget falls
    back to a fixed cut of that sentence only."""
    pieces = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if count_gpt(para) <= size_tokens:
            pieces.append(para)
        else:
            pieces.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())

    chunks, current, current_tok = [], [], 0
    for piece in pieces:
        ptok = count_gpt(piece)
        if ptok > size_tokens:                       # oversized single sentence
            if current:
                chunks.append(" ".join(current)); current, current_tok = [], 0
            chunks.extend(fixed_chunks(piece, size_tokens))
            continue
        if current_tok + ptok > size_tokens and current:
            chunks.append(" ".join(current)); current, current_tok = [], 0
        current.append(piece); current_tok += ptok
    if current:
        chunks.append(" ".join(current))
    return chunks
