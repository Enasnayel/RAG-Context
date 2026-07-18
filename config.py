"""
config.py — single source of truth for every pinned parameter.
Nothing experimental is defined anywhere else. A cell is a (format,
chunking strategy, chunk size) triple; a SCOPE selects which cells run.
"""

# ---- pinned generation controls (must never vary between cells) -----
SEED = 42
GEN_PARAMS = dict(temperature=0, top_p=1, max_tokens=512)
TOP_K = 5

# ---- pinned models ---------------------------------------------------
CONTEXT_MODEL = "claude-haiku-4-5-20251001"   # contextualization snippets
JUDGE_MODEL   = "gpt-4o-mini"                 # faithfulness judge
GPT_MODEL     = "gpt-4o-mini"                 # generator A (tier decision open)
CLAUDE_MODEL  = "claude-haiku-4-5-20251001"   # generator B (tier decision open)
EMBED_MODEL   = "jinaai/jina-embeddings-v2-base-en"  # frozen

# ---- factors ----------------------------------------------------------
FORMATS    = ["json", "toon", "yaml"]
STRATEGIES = ["fixed", "recursive", "semantic"]
SIZES      = [256, 512]

BASELINE = ("json", "fixed", 256)

def cell_id(fmt, strategy, size):
    return f"{fmt}_{strategy}_{size}"

# ---- scopes: which cells a run executes -------------------------------
# pilot = the factorial's first three cells (baseline + primary H1a contrast
# + YAML reference point). core = the full 18-cell factorial. Same code path
# either way.
SCOPES = {
    "pilot": [("json", "fixed", 256), ("toon", "fixed", 256), ("yaml", "fixed", 256)],
    "core":  [(f, s, z) for f in FORMATS for s in STRATEGIES for z in SIZES],
}

# ---- paths -------------------------------------------------------------
DATA_DIR    = "data"
RESULTS_DIR = "results"
CACHE_DIR   = "cache"   # embeddings cached per (corpus, cell) — embed once, ever
