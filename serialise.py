"""
serialise.py — the serialization-format factor (F3).
One dispatch function; every format lives here and nowhere else.
The record ALREADY contains the contextualization snippet in the
"context" field before it reaches this module (Option A order).
"""

import json

FIELD_ORDER = ["doc_id", "title", "chunk_id", "context", "text"]


def serialize(record: dict, fmt: str) -> str:
    if fmt == "json":
        return to_json(record)
    if fmt == "toon":
        return to_toon(record)
    if fmt == "yaml":
        return to_yaml(record)
    raise ValueError(f"unknown format: {fmt}")


# ---- JSON (baseline) --------------------------------------------------
def to_json(record: dict) -> str:
    ordered = {k: record[k] for k in FIELD_ORDER if k in record}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


# ---- TOON (flat records: 'key: value' lines) --------------------------
def _escape_toon(v: str) -> str:
    return v.replace("\\", "\\\\").replace("\n", "\\n")

def _unescape_toon(v: str) -> str:
    out, i = [], 0
    while i < len(v):
        if v[i] == "\\" and i + 1 < len(v):
            out.append("\n" if v[i + 1] == "n" else v[i + 1]); i += 2
        else:
            out.append(v[i]); i += 1
    return "".join(out)

def to_toon(record: dict) -> str:
    return "\n".join(f"{k}: {_escape_toon(str(record[k]))}"
                     for k in FIELD_ORDER if k in record)

def from_toon(text: str) -> dict:
    out = {}
    for line in text.split("\n"):
        if ": " in line:
            k, v = line.split(": ", 1)
            out[k] = _unescape_toon(v)
    return out


# ---- YAML -------------------------------------------------------------
def to_yaml(record: dict) -> str:
    import yaml  # lazy: only needed when yaml cells run
    ordered = {k: record[k] for k in FIELD_ORDER if k in record}
    return yaml.safe_dump(ordered, sort_keys=False,
                          allow_unicode=True, width=10**9).strip()


# ---- lossless validation (required before indexing) --------------------
def roundtrip_ok(record: dict) -> bool:
    ordered = {k: str(record[k]) for k in FIELD_ORDER if k in record}
    return from_toon(to_toon(record)) == ordered
