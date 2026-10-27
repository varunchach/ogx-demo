"""Local extract + token-accurate chunking for BGE (hard max 512).

Robust path:
  1) RecursiveCharacterTextSplitter with length_function = real token count
     (includes special tokens — what the embed server counts).
  2) Hard gate: every chunk ≤ EMBED_MAX_TOKENS before it leaves this module.
  3) Callers should still retry/split on HTTP 400 (tokenizer mismatch).

Do NOT use char*N heuristics as the primary sizing — that caused 513-token 400s.
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Model hard limit is 512. Cap below that so CLS/SEP / vLLM quirks never hit 513.
MODEL_CONTEXT_LIMIT = 512
EMBED_MAX_TOKENS = 500
DEFAULT_HF_TOKENIZER = "BAAI/bge-large-en-v1.5"


def extract_text(filename: str, data: bytes) -> str:
    """Extract plain text from upload bytes (PDF or text)."""
    name = Path(filename).name.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        parts = [(page.extract_text() or "") for page in reader.pages]
        text = "\n".join(parts)
        logger.info(
            "extract pdf pages=%s chars=%s filename=%s",
            len(reader.pages),
            len(text),
            Path(filename).name,
        )
        return text

    text = data.decode("utf-8", errors="ignore")
    logger.info(
        "extract text chars=%s filename=%s", len(text), Path(filename).name
    )
    return text


def resolve_tokenizer_id(embedding_model: str, override: str = "") -> str:
    if (override or "").strip():
        return override.strip()
    mid = (embedding_model or "").lower()
    if "bge" in mid and "small" in mid:
        return "BAAI/bge-small-en-v1.5"
    if "bge" in mid and "base" in mid:
        return "BAAI/bge-base-en-v1.5"
    return DEFAULT_HF_TOKENIZER


@lru_cache(maxsize=4)
def _load_tokenizer(tokenizer_id: str) -> Any:
    from transformers import AutoTokenizer

    logger.info("loading embed tokenizer id=%s", tokenizer_id)
    return AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)


def get_token_counter(
    embedding_model: str = "",
    tokenizer_id: str = "",
) -> tuple[Callable[[str], int], Any, str]:
    """Return (token_len, tokenizer_or_None, mode_label)."""
    tok_id = resolve_tokenizer_id(embedding_model, tokenizer_id)
    try:
        tokenizer = _load_tokenizer(tok_id)

        def _len(text: str) -> int:
            # Count what the model sees (special tokens included).
            return len(tokenizer.encode(text or "", add_special_tokens=True))

        return _len, tokenizer, f"tokenizer:{tok_id}"
    except Exception as exc:  # noqa: BLE001
        logger.exception("tokenizer load failed id=%s: %s", tok_id, exc)

        def _approx(text: str) -> int:
            # Last-resort only — prefer failing closed via hard char cap later.
            return max(1, (len(text or "") + 2) // 2)

        return _approx, None, "chars_fallback"


def truncate_to_tokens(text: str, tokenizer: Any, max_tokens: int) -> str:
    """Truncate so encode(..., add_special_tokens=True) ≤ max_tokens."""
    if tokenizer is None:
        # ~2 chars/token worst-case denseness
        return (text or "")[: max(32, max_tokens * 2)]
    # Reserve room for special tokens by encoding with them, then trim ids.
    ids = tokenizer.encode(text or "", add_special_tokens=True)
    if len(ids) <= max_tokens:
        return text or ""
    # Drop specials, keep content ids under max_tokens - specials budget.
    content = tokenizer.encode(text or "", add_special_tokens=False)
    # Typical BERT: 2 specials (CLS+SEP). Keep content so total ≤ max_tokens.
    specials = max(0, len(ids) - len(content))
    budget = max(1, max_tokens - specials)
    return tokenizer.decode(content[:budget], skip_special_tokens=True)


def ensure_under_limit(
    chunks: list[str],
    token_len: Callable[[str], int],
    tokenizer: Any,
    max_tokens: int = EMBED_MAX_TOKENS,
) -> list[str]:
    """Hard gate: no outgoing chunk may exceed max_tokens."""
    out: list[str] = []
    fixed = 0
    for c in chunks:
        n = token_len(c)
        if n <= max_tokens:
            out.append(c)
            continue
        fixed += 1
        trimmed = truncate_to_tokens(c, tokenizer, max_tokens)
        # If still over (mismatch), keep cutting in half by characters.
        guard = 0
        while token_len(trimmed) > max_tokens and trimmed and guard < 8:
            trimmed = trimmed[: max(32, len(trimmed) * 3 // 4)]
            guard += 1
        if trimmed.strip() and token_len(trimmed) <= max_tokens:
            out.append(trimmed.strip())
        else:
            logger.error(
                "dropped chunk still over limit tokens=%s max=%s preview=%r",
                token_len(trimmed),
                max_tokens,
                (trimmed or "")[:120],
            )
    if fixed:
        logger.warning(
            "ensure_under_limit fixed=%s kept=%s max_tokens=%s",
            fixed,
            len(out),
            max_tokens,
        )
    return out


def split_oversized_chunk(
    text: str,
    token_len: Callable[[str], int],
    tokenizer: Any,
    max_tokens: int = EMBED_MAX_TOKENS,
) -> list[str]:
    """Bisect a chunk that the embed API rejected for context length."""
    text = (text or "").strip()
    if not text:
        return []
    if token_len(text) <= max_tokens:
        return [text]
    # Prefer tokenizer-aware truncate into two halves by content tokens.
    mid = max(1, len(text) // 2)
    # Split on nearest whitespace around midpoint.
    split_at = text.rfind(" ", 0, mid)
    if split_at < mid // 2:
        split_at = mid
    left, right = text[:split_at].strip(), text[split_at:].strip()
    parts: list[str] = []
    for piece in (left, right):
        if not piece:
            continue
        if token_len(piece) <= max_tokens:
            parts.append(piece)
        else:
            parts.extend(
                split_oversized_chunk(piece, token_len, tokenizer, max_tokens)
            )
    return parts or [truncate_to_tokens(text, tokenizer, max_tokens)]


def split_chunks(
    text: str,
    chunk_size_tokens: int = 300,
    chunk_overlap_tokens: int = 64,
    *,
    embedding_model: str = "",
    tokenizer_id: str = "",
    max_tokens: int = EMBED_MAX_TOKENS,
) -> list[str]:
    """Token-sized recursive split + hard cap under model context limit."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Stay strictly under model limit; never request chunk_size ≥ 512.
    target = max(32, min(int(chunk_size_tokens), max_tokens))
    overlap = max(0, min(int(chunk_overlap_tokens), target // 2))
    token_len, tokenizer, mode = get_token_counter(embedding_model, tokenizer_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=target,
        chunk_overlap=overlap,
        length_function=token_len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    raw = [c.strip() for c in splitter.split_text(text or "") if c.strip()]
    chunks = ensure_under_limit(raw, token_len, tokenizer, max_tokens=max_tokens)

    lengths = [token_len(c) for c in chunks] if chunks else []
    logger.info(
        "split mode=%s chunks=%s target=%s/%s hard_max=%s "
        "token_min=%s token_max=%s model_limit=%s",
        mode,
        len(chunks),
        target,
        overlap,
        max_tokens,
        min(lengths) if lengths else 0,
        max(lengths) if lengths else 0,
        MODEL_CONTEXT_LIMIT,
    )
    if lengths and max(lengths) > MODEL_CONTEXT_LIMIT:
        logger.error(
            "BUG: chunk over model limit after gate max=%s", max(lengths)
        )
    return chunks
