"""
contextualise.py — Contextual Retrieval snippets (held constant across cells).

Design facts encoded here:
- Snippets are generated from RAW chunk text (never serialized text), so
  ONE snippet set serves all three formats for a given chunk config.
- The full document rides in a cache_control system block, so chunks of
  the same document hit the Anthropic prompt cache (mandatory cost control).
- Snippets are cached to disk keyed by (corpus, strategy, size): rerunning
  any cell never re-pays contextualization.
"""

import json
from pathlib import Path

from config import CONTEXT_MODEL, CACHE_DIR

_client = None

def _ant():
    global _client
    if _client is None:
        from anthropic import Anthropic  # lazy: only needed when snippets generate
        _client = Anthropic()
    return _client


PROMPT = (
    "Here is a chunk from the document above:\n<chunk>\n{chunk}\n</chunk>\n"
    "Write a short (1-2 sentence) context that situates this chunk within "
    "the overall document, to improve search retrieval of the chunk. "
    "Answer with only the context, nothing else."
)


def snippet_cache_path(corpus: str, strategy: str, size: int) -> Path:
    return Path(CACHE_DIR) / f"snippets_{corpus}_{strategy}_{size}.json"


def contextualize(docs, chunks_by_doc, corpus, strategy, size):
    """Return {(doc_id, chunk_index): snippet}. Disk-cached; safe to rerun."""
    path = snippet_cache_path(corpus, strategy, size)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {tuple(k.rsplit("|", 1)): v for k, v in raw.items()}, True

    path.parent.mkdir(parents=True, exist_ok=True)
    snippets = {}
    for doc in docs:
        doc_id = doc["doc_id"]
        system_block = [{
            "type": "text",
            "text": f"<document>\n{doc['text']}\n</document>",
            "cache_control": {"type": "ephemeral"},
        }]
        for ci, chunk_text in enumerate(chunks_by_doc[doc_id]):
            resp = _ant().messages.create(
                model=CONTEXT_MODEL, max_tokens=150, system=system_block,
                messages=[{"role": "user",
                           "content": PROMPT.format(chunk=chunk_text)}],
            )
            snippets[(doc_id, str(ci))] = resp.content[0].text.strip()

    path.write_text(json.dumps({f"{k[0]}|{k[1]}": v for k, v in snippets.items()},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    return snippets, False
