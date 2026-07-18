"""
tokens.py — token accounting with the GPT tokenizer (cl100k_base).
Lazy-loading: tiktoken downloads its encoding file on first use, so the
import itself never needs network access.
"""

_ENCODER = None


def get_encoder():
    global _ENCODER
    if _ENCODER is None:
        import tiktoken
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_gpt(text: str) -> int:
    """Number of GPT tokens in text."""
    return len(get_encoder().encode(text))
