"""OpenAI-compatible client for cluster OgxServer (OCP AI 3.5).

Ingest (notebook-aligned):
  - local extract + RecursiveCharacterTextSplitter
  - POST /v1/embeddings in small batches
  - POST /v1/vector-io/insert with precomputed vectors

Retrieve:
  - POST /v1/vector-io/query then chat/completions with context
  - fallback: /v1/responses file_search (legacy file-attach stores)

Also: /v1/models, /v1/files, /v1/vector_stores. No direct Postgres.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from src.config import Settings
from src.ingest_chunking import (
    EMBED_MAX_TOKENS,
    ensure_under_limit,
    extract_text,
    get_token_counter,
    split_chunks,
    split_oversized_chunk,
)

logger = logging.getLogger(__name__)


def _preview(text: str, limit: int = 240) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _summarize_attached(attached: dict[str, Any]) -> dict[str, Any]:
    """Compact attach/index payload for logs (no huge blobs)."""
    meta = attached.get("metadata") if isinstance(attached.get("metadata"), dict) else {}
    return {
        "id": attached.get("id"),
        "file_id": attached.get("file_id"),
        "status": attached.get("status"),
        "last_error": attached.get("last_error") or attached.get("error"),
        "chunk_count": attached.get("chunk_count"),
        "chunk_counts": attached.get("chunk_counts"),
        "usage_bytes": attached.get("usage_bytes"),
        "metadata_keys": sorted(meta.keys()) if meta else [],
        "metadata_chunk_count": meta.get("chunk_count"),
        "keys": sorted(str(k) for k in attached.keys()),
    }


def _summarize_responses(data: dict[str, Any]) -> dict[str, Any]:
    """Compact Responses API payload: output types + file_search hit stats."""
    output = data.get("output") or []
    types: list[str] = []
    hit_count = 0
    hit_previews: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        t = str(item.get("type") or "?")
        types.append(t)
        if t not in {"file_search_call", "file_search_tool_call"}:
            continue
        results = item.get("results") or []
        if isinstance(results, dict):
            results = results.get("results") or []
        for hit in results or []:
            if not isinstance(hit, dict):
                continue
            hit_count += 1
            if len(hit_previews) < 3:
                hit_previews.append(
                    {
                        "score": hit.get("score") or hit.get("relevance"),
                        "file_id": hit.get("file_id") or hit.get("filename"),
                        "text": _preview(
                            str(hit.get("text") or hit.get("content") or ""), 180
                        ),
                    }
                )
    return {
        "output_types": types,
        "file_search_hits": hit_count,
        "hit_previews": hit_previews,
        "has_output_text": bool(
            isinstance(data.get("output_text"), str) and data["output_text"].strip()
        ),
        "top_keys": sorted(str(k) for k in data.keys())[:20],
    }


class _ContextTooLong(Exception):
    """Embed API rejected input for exceeding model context length."""


class OgxClient:
    """HTTP client for OgxServer OpenAI-compatible RAG APIs."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base = settings.ogx_base_url.rstrip("/")
        self._model = settings.ogx_model
        self._embed_model = settings.ogx_embedding_model
        self._timeout = settings.ogx_timeout_seconds
        self._headers: dict[str, str] = {}
        if settings.ogx_api_key:
            self._headers["Authorization"] = f"Bearer {settings.ogx_api_key}"
        self._json_headers = {**self._headers, "Content-Type": "application/json"}
        self._vector_store_id: str | None = settings.ogx_vector_store_id or None

    @property
    def configured(self) -> bool:
        return bool(self._base)

    @property
    def vector_store_id(self) -> str | None:
        return self._vector_store_id

    def list_models(self) -> dict[str, Any]:
        self._require()
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{self._base}/v1/models", headers=self._headers)
            resp.raise_for_status()
            return resp.json()

    def ping(self) -> tuple[bool, str]:
        if not self.configured:
            return False, "OGX_BASE_URL unset"
        try:
            self.list_models()
            return True, "ok"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)

    def ensure_vector_store(self) -> str:
        """Return configured or named vector store id (create if missing)."""
        self._require()
        if self._vector_store_id:
            return self._vector_store_id

        name = self._settings.ogx_vector_store_name
        existing = self._find_vector_store_by_name(name)
        if existing:
            self._vector_store_id = existing
            logger.info("Reusing OGX vector store id=%s name=%s", existing, name)
            return existing

        # Notebook-style create: embedding_model + dimension + provider_id.
        # Set OGX_EMBEDDING_DIMENSION=0 to omit dim (granite matryoshka 400).
        payload: dict[str, Any] = {"name": name}
        if self._embed_model:
            payload["embedding_model"] = self._embed_model
        if self._settings.ogx_vector_provider_id:
            payload["provider_id"] = self._settings.ogx_vector_provider_id
        dim = int(self._settings.ogx_embedding_dimension or 0)
        if dim > 0:
            payload["embedding_dimension"] = dim

        logger.info("Creating OGX vector store name=%s payload=%s", name, payload)
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base}/v1/vector_stores",
                headers=self._json_headers,
                json=payload,
            )
            if resp.status_code >= 400:
                alt = {
                    "name": name,
                    "embedding_model": self._embed_model,
                    "provider_id": self._settings.ogx_vector_provider_id,
                }
                if dim > 0:
                    alt["embedding_dimension"] = dim
                logger.warning(
                    "Vector store create HTTP %s body=%r; retry",
                    resp.status_code,
                    _preview(resp.text, 300),
                )
                resp = client.post(
                    f"{self._base}/v1/vector_stores",
                    headers=self._json_headers,
                    json=alt,
                )
            resp.raise_for_status()
            data = resp.json()
        store_id = data.get("id") or data.get("vector_store_id")
        if not store_id:
            raise RuntimeError(f"Vector store create returned no id: {data!r}")
        self._vector_store_id = str(store_id)
        logger.info("Created OGX vector store id=%s", self._vector_store_id)
        return self._vector_store_id

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """POST /v1/embeddings — one batch. On 400 context-length, split & retry."""
        self._require()
        if not texts:
            return []
        if not self._embed_model:
            raise RuntimeError("OGX_EMBEDDING_MODEL is not set")

        # Single-string path used by retry recursion.
        def _embed_once(batch: list[str]) -> list[list[float]]:
            payload: dict[str, Any] = {"model": self._embed_model, "input": batch}
            t0 = time.perf_counter()
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._base}/v1/embeddings",
                    headers=self._json_headers,
                    json=payload,
                )
            ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code >= 400:
                body = resp.text or ""
                logger.error(
                    "OGX embeddings HTTP %s n=%s ms=%.0f body=%r",
                    resp.status_code,
                    len(batch),
                    ms,
                    _preview(body, 500),
                )
                # Context overflow: split each item and retry (don't fail whole doc).
                if resp.status_code == 400 and (
                    "maximum context length" in body
                    or "input_tokens" in body
                    or "512" in body
                ):
                    raise _ContextTooLong(body)
                resp.raise_for_status()
            data = resp.json()
            rows = data.get("data") or []
            vectors = [list(row["embedding"]) for row in rows]
            logger.info(
                "OGX embeddings ok n=%s dim=%s ms=%.0f",
                len(vectors),
                len(vectors[0]) if vectors else 0,
                ms,
            )
            return vectors

        try:
            return _embed_once(texts)
        except _ContextTooLong:
            if len(texts) > 1:
                # Isolate the offender(s): embed one-by-one.
                out: list[list[float]] = []
                for t in texts:
                    out.extend(self.create_embeddings([t]))
                return out
            # Single oversized string — bisect until it fits.
            token_len, tokenizer, _mode = get_token_counter(
                self._embed_model,
                getattr(self._settings, "ogx_embedding_tokenizer", "") or "",
            )
            parts = split_oversized_chunk(
                texts[0], token_len, tokenizer, max_tokens=EMBED_MAX_TOKENS
            )
            parts = ensure_under_limit(
                parts, token_len, tokenizer, max_tokens=EMBED_MAX_TOKENS
            )
            logger.warning(
                "embed context overflow; split 1 chunk → %s pieces", len(parts)
            )
            if not parts:
                raise RuntimeError("chunk empty after context-length split")
            if len(parts) == 1 and parts[0] == texts[0]:
                raise RuntimeError(
                    "chunk still over embed context limit after split"
                )
            out: list[list[float]] = []
            for p in parts:
                out.extend(self.create_embeddings([p]))
            # Caller expects 1 vector per input text; average-pool pieces.
            if len(out) == 1:
                return out
            dim = len(out[0])
            pooled = [sum(v[i] for v in out) / len(out) for i in range(dim)]
            return [pooled]

    def vector_io_insert(
        self,
        store_id: str,
        chunks: list[str],
        embeddings: list[list[float]],
        source: str,
        doc_id: str,
    ) -> None:
        """POST /v1/vector-io/insert with precomputed embeddings (notebook path)."""
        self._require()
        if len(chunks) != len(embeddings):
            raise RuntimeError(
                f"chunk/embedding length mismatch {len(chunks)}!={len(embeddings)}"
            )
        dim = int(self._settings.ogx_embedding_dimension or 0)
        if dim <= 0 and embeddings:
            dim = len(embeddings[0])
        payload_chunks: list[dict[str, Any]] = []
        for i, (content, emb) in enumerate(zip(chunks, embeddings, strict=True)):
            chunk_id = f"{doc_id}-{i}"
            payload_chunks.append(
                {
                    "content": content,
                    "chunk_id": chunk_id,
                    "metadata": {
                        "document_id": doc_id,
                        "source": source,
                        "chunk_index": i,
                    },
                    "chunk_metadata": {
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                    },
                    "embedding": emb,
                    "embedding_model": self._embed_model,
                    "embedding_dimension": dim,
                }
            )
        body: dict[str, Any] = {
            "vector_store_id": store_id,
            "chunks": payload_chunks,
        }
        paths = ("/v1/vector-io/insert", "/v1/vector_io/insert")
        last_err = ""
        with httpx.Client(timeout=self._timeout) as client:
            for path in paths:
                resp = client.post(
                    f"{self._base}{path}",
                    headers=self._json_headers,
                    json=body,
                )
                if resp.status_code < 400:
                    logger.info(
                        "OGX vector_io.insert via=%s store=%s n=%s",
                        path,
                        store_id,
                        len(payload_chunks),
                    )
                    return
                last_err = f"{path} HTTP {resp.status_code} {_preview(resp.text, 400)}"
                logger.warning("OGX vector_io.insert failed %s", last_err)
        raise RuntimeError(f"vector_io.insert failed: {last_err}")

    def vector_io_query(
        self,
        store_id: str,
        query: str,
        max_chunks: int = 5,
    ) -> list[dict[str, Any]]:
        """POST /v1/vector-io/query — returns [{content, score, metadata}, ...]."""
        self._require()
        body: dict[str, Any] = {
            "vector_store_id": store_id,
            "query": query,
            "params": {"max_chunks": max_chunks},
        }
        paths = ("/v1/vector-io/query", "/v1/vector_io/query")
        last_err = ""
        with httpx.Client(timeout=self._timeout) as client:
            for path in paths:
                resp = client.post(
                    f"{self._base}{path}",
                    headers=self._json_headers,
                    json=body,
                )
                if resp.status_code >= 400:
                    last_err = (
                        f"{path} HTTP {resp.status_code} {_preview(resp.text, 300)}"
                    )
                    continue
                data = resp.json()
                chunks = data.get("chunks") or data.get("data") or []
                scores = data.get("scores") or []
                sources: list[dict[str, Any]] = []
                for i, ch in enumerate(chunks):
                    if isinstance(ch, str):
                        content = ch
                        meta: dict[str, Any] = {}
                    elif isinstance(ch, dict):
                        content = str(
                            ch.get("content") or ch.get("text") or ch.get("chunk") or ""
                        )
                        meta = (
                            ch.get("metadata")
                            if isinstance(ch.get("metadata"), dict)
                            else {}
                        )
                    else:
                        content = str(ch)
                        meta = {}
                    score = float(scores[i]) if i < len(scores) else 0.0
                    sources.append(
                        {"content": content, "score": score, "metadata": meta}
                    )
                logger.info(
                    "OGX vector_io.query via=%s store=%s hits=%s",
                    path,
                    store_id,
                    len(sources),
                )
                return sources
        raise RuntimeError(f"vector_io.query failed: {last_err}")

    def upload_file(self, filename: str, data: bytes) -> str:
        """POST /v1/files — OGX Files API."""
        self._require()
        name = Path(filename).name
        files = {"file": (name, data)}
        form = {"purpose": "assistants"}
        logger.info("OGX upload file name=%s bytes=%s", name, len(data))
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base}/v1/files",
                headers=self._headers,
                files=files,
                data=form,
            )
            resp.raise_for_status()
            body = resp.json()
        file_id = body.get("id")
        if not file_id:
            raise RuntimeError(f"File upload returned no id: {body!r}")
        return str(file_id)

    def attach_file_to_vector_store(self, vector_store_id: str, file_id: str) -> dict[str, Any]:
        """Attach file; OGX chunks/embeds/indexes into its pgvector provider."""
        self._require()
        payload: dict[str, Any] = {
            "file_id": file_id,
            "chunking_strategy": {
                "type": "static",
                "static": {
                    "max_chunk_size_tokens": self._settings.chunk_size_tokens,
                    "chunk_overlap_tokens": self._settings.chunk_overlap_tokens,
                },
            },
        }
        url = f"{self._base}/v1/vector_stores/{vector_store_id}/files"
        logger.info(
            "OGX attach file_id=%s store=%s chunk=%s/%s",
            file_id,
            vector_store_id,
            self._settings.chunk_size_tokens,
            self._settings.chunk_overlap_tokens,
        )
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(url, headers=self._json_headers, json=payload)
            resp.raise_for_status()
            attached = resp.json()

        # Poll until indexed (status names vary slightly by build)
        status = str(attached.get("status") or "").lower()
        vs_file_id = attached.get("id") or file_id
        logger.info(
            "OGX attach poll start store=%s vs_file=%s initial_status=%s last_error=%s",
            vector_store_id,
            vs_file_id,
            status or "(empty)",
            attached.get("last_error") or attached.get("error"),
        )
        deadline = time.time() + min(self._timeout, 170)
        polls = 0
        while status in {"in_progress", "pending", "processing", ""} and time.time() < deadline:
            time.sleep(1.5)
            polls += 1
            with httpx.Client(timeout=30.0) as client:
                poll = client.get(
                    f"{self._base}/v1/vector_stores/{vector_store_id}/files/{vs_file_id}",
                    headers=self._headers,
                )
                if poll.status_code >= 400:
                    logger.error(
                        "OGX attach poll HTTP %s body=%r",
                        poll.status_code,
                        _preview(poll.text, 500),
                    )
                    break
                attached = poll.json()
                status = str(attached.get("status") or "").lower()
                if polls <= 3 or polls % 5 == 0:
                    logger.info(
                        "OGX attach poll #%s status=%s last_error=%s usage_bytes=%s",
                        polls,
                        status or "(empty)",
                        attached.get("last_error") or attached.get("error"),
                        attached.get("usage_bytes"),
                    )
                if status in {"completed", "ready", "active"}:
                    break
                if status in {"failed", "cancelled", "error"}:
                    err = attached.get("last_error") or attached.get("error")
                    logger.error(
                        "OGX INDEX FAILED store=%s file=%s status=%s last_error=%s "
                        "full=%s",
                        vector_store_id,
                        vs_file_id,
                        status,
                        err,
                        _summarize_attached(attached),
                    )
                    raise RuntimeError(
                        f"Vector store file indexing failed status={status} "
                        f"last_error={err!r}"
                    )
        if status not in {"completed", "ready", "active"}:
            err = attached.get("last_error") or attached.get("error")
            logger.error(
                "OGX attach ended WITHOUT completed status=%s polls=%s last_error=%s",
                status,
                polls,
                err,
            )
        else:
            logger.info(
                "OGX attach SUCCESS store=%s file=%s status=%s polls=%s",
                vector_store_id,
                vs_file_id,
                status,
                polls,
            )
        return attached

    def _log_store_inventory(self, store_id: str, label: str) -> list[dict[str, Any]]:
        """Log every file status in the store (critical for zero-hit debugging)."""
        files = self.list_vector_store_files(store_id)
        rows: list[dict[str, Any]] = []
        for f in files[:30]:
            if not isinstance(f, dict):
                continue
            rows.append(
                {
                    "id": f.get("id"),
                    "file_id": f.get("file_id"),
                    "status": f.get("status"),
                    "usage_bytes": f.get("usage_bytes"),
                    "last_error": f.get("last_error") or f.get("error"),
                    "chunk_count": f.get("chunk_count"),
                }
            )
        completed = sum(
            1
            for r in rows
            if str(r.get("status") or "").lower() in {"completed", "ready", "active"}
        )
        failed = sum(
            1
            for r in rows
            if str(r.get("status") or "").lower() in {"failed", "cancelled", "error"}
        )
        logger.info(
            "OGX store inventory [%s] store=%s files=%s completed=%s failed=%s "
            "embed_model=%s rows=%s",
            label,
            store_id,
            len(files),
            completed,
            failed,
            self._embed_model,
            rows,
        )
        return files

    def delete_vector_store_file(self, vector_store_id: str, vs_file_id: str) -> bool:
        """DELETE /v1/vector_stores/{id}/files/{file_id} — best-effort."""
        self._require()
        url = f"{self._base}/v1/vector_stores/{vector_store_id}/files/{vs_file_id}"
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.delete(url, headers=self._headers)
            if resp.status_code in {200, 204, 404}:
                logger.info(
                    "OGX dedupe deleted store_file=%s store=%s http=%s",
                    vs_file_id,
                    vector_store_id,
                    resp.status_code,
                )
                return True
            logger.warning(
                "OGX dedupe delete failed store_file=%s http=%s body=%r",
                vs_file_id,
                resp.status_code,
                _preview(resp.text, 300),
            )
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("OGX dedupe delete error store_file=%s: %s", vs_file_id, exc)
            return False

    def _resolve_store_file_name(self, entry: dict[str, Any]) -> str:
        """Best-effort filename for a vector-store file entry."""
        for key in ("filename", "file_name", "name"):
            val = entry.get(key)
            if isinstance(val, str) and val.strip():
                return Path(val.strip()).name
        meta = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        attrs = entry.get("attributes") if isinstance(entry.get("attributes"), dict) else {}
        for blob in (meta, attrs):
            for key in ("filename", "file_name", "name", "original_filename"):
                val = blob.get(key)
                if isinstance(val, str) and val.strip():
                    return Path(val.strip()).name
        file_id = str(entry.get("file_id") or entry.get("id") or "")
        if not file_id:
            return ""
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.get(
                    f"{self._base}/v1/files/{file_id}",
                    headers=self._headers,
                )
            if resp.status_code >= 400:
                return ""
            body = resp.json() if resp.content else {}
            for key in ("filename", "file_name", "name"):
                val = body.get(key)
                if isinstance(val, str) and val.strip():
                    return Path(val.strip()).name
        except Exception:  # noqa: BLE001
            logger.debug("Could not resolve filename for file_id=%s", file_id)
        return ""

    def remove_store_files_named(self, vector_store_id: str, filename: str) -> int:
        """Delete all store files whose basename matches (case-insensitive)."""
        want = Path(filename).name.casefold()
        if not want:
            return 0
        removed = 0
        for entry in self.list_vector_store_files(vector_store_id):
            if not isinstance(entry, dict):
                continue
            name = self._resolve_store_file_name(entry)
            if not name or name.casefold() != want:
                continue
            vs_id = str(entry.get("id") or entry.get("file_id") or "")
            if not vs_id:
                continue
            if self.delete_vector_store_file(vector_store_id, vs_id):
                removed += 1
        if removed:
            logger.info(
                "OGX dedupe removed %s prior file(s) named=%r from store=%s",
                removed,
                Path(filename).name,
                vector_store_id,
            )
        return removed

    def ingest_bytes(self, filename: str, data: bytes) -> dict[str, Any]:
        """Notebook-aligned ingest: local chunk → OGX embed batches → vector_io.insert.

        Does not use file-attach auto-embed (that path 502'd on large PDFs when
        BGE queued ~80 reqs with Running=0).
        """
        name = Path(filename).name
        batch_size = max(1, int(self._settings.ogx_embed_batch_size or 16))
        logger.info(
            "OGX ingest START filename=%s bytes=%s embed_model=%s "
            "store_name=%s provider=%s batch=%s chunk=%s/%s",
            name,
            len(data),
            self._embed_model,
            self._settings.ogx_vector_store_name,
            self._settings.ogx_vector_provider_id,
            batch_size,
            self._settings.chunk_size_tokens,
            self._settings.chunk_overlap_tokens,
        )
        text = extract_text(filename, data)
        if not (text or "").strip():
            raise RuntimeError(f"No extractable text from {name}")
        tok_id = getattr(self._settings, "ogx_embedding_tokenizer", "") or ""
        chunks = split_chunks(
            text,
            chunk_size_tokens=self._settings.chunk_size_tokens,
            chunk_overlap_tokens=self._settings.chunk_overlap_tokens,
            embedding_model=self._embed_model,
            tokenizer_id=tok_id,
            max_tokens=EMBED_MAX_TOKENS,
        )
        if not chunks:
            raise RuntimeError(f"Chunking produced 0 chunks for {name}")

        # Final gate (same tokenizer) before any HTTP embed call.
        token_len, tokenizer, _mode = get_token_counter(self._embed_model, tok_id)
        chunks = ensure_under_limit(
            chunks, token_len, tokenizer, max_tokens=EMBED_MAX_TOKENS
        )
        if not chunks:
            raise RuntimeError(f"No chunks left after token limit gate for {name}")

        store_id = self.ensure_vector_store()
        removed = 0
        if self._settings.ogx_ingest_dedupe:
            # Best-effort: remove prior Files API entries with same name (legacy).
            removed = self.remove_store_files_named(store_id, filename)

        embeddings: list[list[float]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            logger.info(
                "OGX ingest embed batch %s-%s / %s",
                start + 1,
                start + len(batch),
                len(chunks),
            )
            # Batch first; on 400 context-length, create_embeddings isolates/splits.
            embeddings.extend(self.create_embeddings(batch))
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"embed count {len(embeddings)} != chunk count {len(chunks)}"
            )

        doc_id = f"doc-{uuid.uuid4().hex[:8]}"
        self.vector_io_insert(
            store_id=store_id,
            chunks=chunks,
            embeddings=embeddings,
            source=name,
            doc_id=doc_id,
        )
        logger.info(
            "OGX ingest done filename=%s store=%s chunks=%s dim=%s doc_id=%s "
            "deduped_removed=%s",
            name,
            store_id,
            len(chunks),
            len(embeddings[0]) if embeddings else 0,
            doc_id,
            removed,
        )
        return {
            "filename": name,
            "file_id": doc_id,
            "vector_store_id": store_id,
            "chunk_count": len(chunks),
            "status": "completed",
            "collection": store_id,
            "deduped_removed": removed,
        }

    def list_vector_store_files(self, vector_store_id: str | None = None) -> list[dict[str, Any]]:
        store_id = vector_store_id or self.ensure_vector_store()
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(
                f"{self._base}/v1/vector_stores/{store_id}/files",
                headers=self._headers,
            )
            if resp.status_code >= 400:
                return []
            data = resp.json()
        return list(data.get("data") or data.get("files") or [])

    def file_count(self) -> int:
        try:
            if not self._vector_store_id and not self._settings.ogx_vector_store_id:
                # Avoid creating a store on health checks
                found = self._find_vector_store_by_name(self._settings.ogx_vector_store_name)
                if not found:
                    return 0
                self._vector_store_id = found
            return len(self.list_vector_store_files())
        except Exception:  # noqa: BLE001
            logger.exception("file_count failed")
            return 0

    def rag_chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """RAG via Responses + file_search when available; else chat/completions."""
        self._require()
        # Greetings must not force retrieval — that produced "meeting minutes"
        # answers for "Hi" when prior turns were still in history.
        # ogx-server-new often has no /v1/responses → use chat/completions.
        if _is_smalltalk(message):
            answer = self.chat(
                message,
                history=history,
                system=(
                    "You are a friendly assistant for a document Q&A demo. "
                    "Greet the user briefly. Do not invent document content."
                ),
            )
            return _strip_control_tokens(answer), []

        store_id = self.ensure_vector_store()
        logger.info(
            "OGX rag_chat start store=%s store_name=%s model=%s embed=%s "
            "msg_len=%s history_turns=%s msg_preview=%r",
            store_id,
            self._settings.ogx_vector_store_name,
            self._model,
            self._embed_model,
            len(message or ""),
            len(history or []),
            _preview(message, 160),
        )

        # Primary: vector_io.query (matches notebook ingest via vector_io.insert).
        try:
            sources = self.vector_io_query(store_id, message, max_chunks=5)
            if sources:
                context = "\n\n".join(
                    f"[{i + 1}] {s.get('content') or ''}"
                    for i, s in enumerate(sources)
                )
                system = (
                    "You are a helpful RAG assistant. Answer using ONLY the "
                    "document excerpts below. If they do not contain the answer, "
                    "say you do not have that information in the documents. "
                    "Do not invent facts.\n\n"
                    f"Document excerpts:\n{context}"
                )
                answer = self.chat(message, history=history, system=system)
                logger.info(
                    "OGX rag_chat done via=vector_io.query store=%s hits=%s "
                    "answer_preview=%r",
                    store_id,
                    len(sources),
                    _preview(answer, 200),
                )
                return _strip_control_tokens(answer), sources
            logger.warning(
                "OGX vector_io.query returned 0 hits store=%s — trying file_search",
                store_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "vector_io.query failed (%s); trying file_search", exc, exc_info=True
            )

        try:
            answer, sources = self._responses_file_search(
                message, store_id, history or []
            )
            logger.info(
                "OGX rag_chat done via=file_search store=%s retrieve_hits=%s "
                "answer_preview=%r",
                store_id,
                len(sources),
                _preview(answer, 200),
            )
            return answer, sources
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "file_search failed (%s); falling back to plain chat",
                exc,
                exc_info=True,
            )
            answer = self._chat_with_hint(message, store_id, history or [])
            return answer, []

    def chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        system: str | None = None,
    ) -> str:
        """POST /v1/chat/completions (ogx-server-new / classic OpenAI chat)."""
        self._require()
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(history or [])
        messages.append({"role": "user", "content": message})
        payload: dict[str, Any] = {"messages": messages}
        if self._model:
            payload["model"] = self._model
        logger.info(
            "OGX chat/completions model=%s msg_preview=%r",
            self._model,
            _preview(message, 160),
        )
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base}/v1/chat/completions",
                headers=self._json_headers,
                json=payload,
            )
            if resp.status_code >= 400:
                logger.error(
                    "OGX chat/completions HTTP %s body=%r",
                    resp.status_code,
                    _preview(resp.text, 500),
                )
            resp.raise_for_status()
            data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected OGX chat response: {data!r}") from exc

    def _responses_file_search(
        self,
        message: str,
        store_id: str,
        history: list[dict[str, str]],
    ) -> tuple[str, list[dict[str, Any]]]:
        # CRITICAL: file_search embeds `input`. Never put prior chat into input —
        # that made turn-2+ retrieval fail (hits 9 then hits 0). Search query =
        # current user message only. Brief history goes in instructions only.
        prior_note = ""
        if history:
            prior = "\n".join(
                f"{m.get('role', 'user')}: {(m.get('content') or '')[:200]}"
                for m in history[-2:]
            )
            prior_note = (
                f" Recent chat (do not search on this; answer the input "
                f"question only):\n{prior}\n"
            )

        instructions = (
            "You are a helpful RAG assistant. "
            "Use file_search on the user input question only. "
            "Answer using ONLY retrieved document chunks. "
            "If chunks do not contain the answer, say you do not have that "
            "information in the documents. Do not invent facts. "
            "Do not emit tool/citation control tokens such as <|...|>."
            f"{prior_note}"
        )

        payload: dict[str, Any] = {
            "input": message,
            "instructions": instructions,
            "tools": [
                {
                    "type": "file_search",
                    "vector_store_ids": [store_id],
                }
            ],
            # Force retrieval for non-smalltalk questions (smalltalk is
            # short-circuited earlier in rag_chat).
            "tool_choice": {"type": "file_search"},
            "include": ["file_search_call.results"],
        }
        if self._model:
            payload["model"] = self._model

        logger.info(
            "OGX responses file_search POST store=%s model=%s tool_choice=file_search "
            "input_preview=%r history_in_instructions=%s",
            store_id,
            self._model,
            _preview(message, 200),
            bool(history),
        )
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base}/v1/responses",
                headers=self._json_headers,
                json=payload,
            )
            logger.info(
                "OGX responses HTTP status=%s bytes=%s",
                resp.status_code,
                len(resp.content or b""),
            )
            if resp.status_code >= 400:
                logger.error(
                    "OGX responses error body_preview=%r",
                    _preview(resp.text, 800),
                )
            resp.raise_for_status()
            data = resp.json()

        summary = _summarize_responses(data if isinstance(data, dict) else {})
        logger.info("OGX responses summary=%s", summary)

        answer = _strip_control_tokens(_extract_response_text(data))
        sources = _extract_file_search_sources(data)
        logger.info(
            "OGX responses parsed answer_len=%s sources=%s "
            "source0_preview=%r",
            len(answer or ""),
            len(sources),
            _preview(sources[0].get("content", ""), 180) if sources else "",
        )
        if not answer:
            raise RuntimeError(f"Empty Responses payload: {data!r}")
        return answer, sources

    def debug_rag(self, message: str) -> dict[str, Any]:
        """Stage-by-stage RAG diagnosis for /debug/rag.

        Stages: ogx → models → embed_api → store → files → file_search → answer.
        Each stage is pass/fail so you can see exactly where the pipeline breaks.
        """
        self._require()
        stages: list[dict[str, Any]] = []

        def _stage(name: str, ok: bool, detail: Any) -> None:
            stages.append({"stage": name, "ok": ok, "detail": detail})
            logger.info("debug/rag stage=%s ok=%s detail=%s", name, ok, detail)

        # --- 1) OGX reachable ---
        ping_ok, ping_msg = self.ping()
        _stage("1_ogx_reachable", ping_ok, ping_msg)
        if not ping_ok:
            return {
                "stages": stages,
                "failed_at": "1_ogx_reachable",
                "diagnosis": f"FAIL: cannot reach OGX ({ping_msg})",
            }

        # --- 2) Models listed; LLM + embed ids present ---
        models_payload: dict[str, Any] = {}
        model_ids: list[str] = []
        try:
            models_payload = self.list_models()
            for m in models_payload.get("data") or []:
                mid = m.get("id") if isinstance(m, dict) else None
                if mid:
                    model_ids.append(str(mid))
        except Exception as exc:  # noqa: BLE001
            _stage("2_list_models", False, str(exc))
            return {
                "stages": stages,
                "failed_at": "2_list_models",
                "diagnosis": f"FAIL: /v1/models error: {exc}",
            }

        llm_ok = bool(self._model) and (
            self._model in model_ids or any(self._model in x for x in model_ids)
        )
        embed_ok = bool(self._embed_model) and (
            self._embed_model in model_ids
            or any(self._embed_model in x for x in model_ids)
        )
        _stage(
            "2_llm_model_listed",
            llm_ok,
            {"configured": self._model, "listed": llm_ok, "sample_ids": model_ids[:30]},
        )
        _stage(
            "3_embedding_model_listed",
            embed_ok,
            {"configured": self._embed_model, "listed": embed_ok},
        )

        # --- 3) Direct embed probe (if OGX exposes /v1/embeddings) ---
        embed_probe: dict[str, Any] = {"attempted": False}
        try:
            embed_probe["attempted"] = True
            with httpx.Client(timeout=60.0) as client:
                er = client.post(
                    f"{self._base}/v1/embeddings",
                    headers=self._json_headers,
                    json={
                        "model": self._embed_model,
                        "input": "rag debug embedding probe",
                    },
                )
            embed_probe["http_status"] = er.status_code
            if er.status_code < 400:
                ej = er.json()
                vec = None
                try:
                    vec = ej["data"][0]["embedding"]
                except (KeyError, IndexError, TypeError):
                    pass
                dim = len(vec) if isinstance(vec, list) else None
                embed_probe["vector_dim"] = dim
                _stage(
                    "4_embedding_api",
                    bool(dim),
                    embed_probe,
                )
            else:
                embed_probe["body_preview"] = _preview(er.text, 400)
                # 404 = OGX may only embed inside vector_stores — not a hard fail
                soft = er.status_code in {404, 405}
                _stage(
                    "4_embedding_api",
                    soft,
                    {
                        **embed_probe,
                        "note": (
                            "endpoint missing — OGX may embed only on attach; "
                            "not conclusive"
                            if soft
                            else "embed API failed"
                        ),
                    },
                )
        except Exception as exc:  # noqa: BLE001
            _stage(
                "4_embedding_api",
                True,
                {
                    "attempted": True,
                    "error": str(exc),
                    "note": "probe failed; OGX may not expose /v1/embeddings",
                },
            )

        # --- 4) Vector store ---
        store_id = self.ensure_vector_store()
        store_meta: dict[str, Any] = {"id": store_id}
        try:
            with httpx.Client(timeout=30.0) as client:
                sr = client.get(
                    f"{self._base}/v1/vector_stores/{store_id}",
                    headers=self._headers,
                )
            store_meta["http_status"] = sr.status_code
            if sr.status_code < 400:
                body = sr.json()
                store_meta["name"] = body.get("name")
                store_meta["status"] = body.get("status")
                store_meta["file_counts"] = body.get("file_counts")
                store_meta["metadata"] = body.get("metadata")
                store_meta["keys"] = sorted(str(k) for k in body.keys())[:30]
            else:
                store_meta["body_preview"] = _preview(sr.text, 300)
        except Exception as exc:  # noqa: BLE001
            store_meta["error"] = str(exc)
        _stage("5_vector_store", bool(store_id), store_meta)

        # --- 5) Files in store ---
        files = self.list_vector_store_files(store_id)
        file_summaries: list[dict[str, Any]] = []
        completed = 0
        for f in files[:20]:
            if not isinstance(f, dict):
                continue
            chunks = f.get("chunk_count")
            if chunks is None and isinstance(f.get("chunk_counts"), dict):
                chunks = f["chunk_counts"].get("total_chunks")
            st = str(f.get("status") or "").lower()
            if st in {"completed", "ready", "active"}:
                completed += 1
            file_summaries.append(
                {
                    "id": f.get("id"),
                    "file_id": f.get("file_id"),
                    "status": f.get("status"),
                    "chunk_count": chunks,
                    "usage_bytes": f.get("usage_bytes"),
                    "last_error": f.get("last_error") or f.get("error"),
                }
            )
        files_ok = len(files) > 0 and completed > 0
        _stage(
            "6_files_indexed",
            files_ok,
            {
                "files_count": len(files),
                "completed_count": completed,
                "files": file_summaries,
                "hint": (
                    None
                    if files_ok
                    else "No completed files in store — re-upload after fixing embed"
                ),
            },
        )

        # --- 6) file_search ---
        payload: dict[str, Any] = {
            "input": message,
            "instructions": (
                "Answer using ONLY file_search results. If none, say so."
            ),
            "tools": [
                {
                    "type": "file_search",
                    "vector_store_ids": [store_id],
                }
            ],
            "tool_choice": {"type": "file_search"},
            "include": ["file_search_call.results"],
        }
        if self._model:
            payload["model"] = self._model

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base}/v1/responses",
                headers=self._json_headers,
                json=payload,
            )
            body_text = resp.text
            try:
                data = resp.json()
            except Exception:  # noqa: BLE001
                data = {"_raw": body_text[:4000]}

        summary = (
            _summarize_responses(data) if isinstance(data, dict) else {"raw": True}
        )
        answer = (
            _strip_control_tokens(_extract_response_text(data))
            if isinstance(data, dict)
            else ""
        )
        sources = (
            _extract_file_search_sources(data) if isinstance(data, dict) else []
        )
        search_ok = resp.status_code < 400 and len(sources) > 0
        _stage(
            "7_file_search",
            search_ok,
            {
                "http_status": resp.status_code,
                "retrieve_hits": len(sources),
                "responses_summary": summary,
                "sources_preview": [
                    {
                        "score": s.get("score"),
                        "text": _preview(str(s.get("content") or ""), 160),
                    }
                    for s in sources[:3]
                ],
            },
        )

        # --- 7) Answer ---
        _stage(
            "8_answer",
            bool(answer) and len(sources) > 0,
            {"answer_preview": _preview(answer, 400), "answer_len": len(answer or "")},
        )

        failed_at = next((s["stage"] for s in stages if not s["ok"]), None)
        if failed_at is None:
            diagnosis = "OK: all stages passed (embed listed, files indexed, file_search hits)"
        elif failed_at == "3_embedding_model_listed":
            diagnosis = (
                "FAIL at embedding model: configured id not in /v1/models — "
                "fix OGX_EMBEDDING_MODEL to an id from sample_ids"
            )
        elif failed_at == "4_embedding_api":
            diagnosis = (
                "FAIL at embedding API: probe failed "
                f"(vector_dim={embed_probe.get('vector_dim')})"
            )
        elif failed_at == "6_files_indexed":
            diagnosis = (
                "FAIL at indexing: store has no completed files "
                "(upload/attach/embed into pgvector never finished)"
            )
        elif failed_at == "7_file_search":
            diagnosis = (
                "FAIL at file_search: files exist but ZERO hits — "
                "wrong embed model for this store, empty PDF text, "
                "or search index broken. Recreate store + re-ingest."
            )
        else:
            diagnosis = f"FAIL at {failed_at}"

        return {
            "query": message,
            "ogx_base_url": self._base,
            "model": self._model,
            "embedding_model": self._embed_model,
            "vector_store_id": store_id,
            "vector_store_name": self._settings.ogx_vector_store_name,
            "stages": stages,
            "failed_at": failed_at,
            "diagnosis": diagnosis,
            "retrieve_hits": len(sources),
            "sources": sources,
            "answer_preview": _preview(answer, 400),
        }

    def _chat_with_hint(
        self,
        message: str,
        store_id: str,
        history: list[dict[str, str]],
    ) -> str:
        system = (
            "You are a helpful assistant. Use knowledge from the project's "
            f"OGX vector store ({store_id}) when relevant. If you are unsure, say so."
        )
        return self.chat(message, history=history, system=system)

    def _find_vector_store_by_name(self, name: str) -> str | None:
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(f"{self._base}/v1/vector_stores", headers=self._headers)
            if resp.status_code >= 400:
                return None
            data = resp.json()
        for item in data.get("data") or data.get("vector_stores") or []:
            if item.get("name") == name and item.get("id"):
                return str(item["id"])
        return None

    def _require(self) -> None:
        if not self.configured:
            raise RuntimeError("OGX_BASE_URL is not set")


_SMALLTALK_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank you|thx|good morning|good afternoon|"
    r"good evening|howdy|sup|what's up|whats up)\s*[!.?]*\s*$",
    re.IGNORECASE,
)


def _is_smalltalk(message: str) -> bool:
    return bool(_SMALLTALK_RE.match((message or "").strip()))


def _strip_control_tokens(text: str) -> str:
    """Remove leaked tool/citation control tokens like <|uuid|> from model output."""
    if not text:
        return text
    cleaned = re.sub(r"<\|[^|>]{1,128}\|>", "", text)
    return cleaned.strip()


def _extract_chunk_count(attached: dict[str, Any]) -> int:
    for key in ("chunk_count", "chunk_counts", "usage_bytes"):
        val = attached.get(key)
        if isinstance(val, int) and key == "chunk_count":
            return val
        if isinstance(val, dict) and "total_chunks" in val:
            try:
                return int(val["total_chunks"])
            except (TypeError, ValueError):
                pass
    meta = attached.get("metadata") or {}
    if isinstance(meta, dict) and "chunk_count" in meta:
        try:
            return int(meta["chunk_count"])
        except (TypeError, ValueError):
            pass
    # OGX may not expose chunk count; 1 = file attached successfully
    return 1 if attached.get("id") or attached.get("file_id") else 0


def _extract_response_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"]
    # OpenAI Responses shape: output[] -> message -> content[] -> output_text
    parts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            for block in item.get("content") or []:
                if isinstance(block, dict) and block.get("type") in {
                    "output_text",
                    "text",
                }:
                    text = block.get("text") or block.get("output_text") or ""
                    if text:
                        parts.append(str(text))
        if item.get("type") == "output_text" and item.get("text"):
            parts.append(str(item["text"]))
    if parts:
        return "\n".join(parts)
    # Rare fallback shapes
    if data.get("choices"):
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass
    return ""


def _extract_file_search_sources(data: dict[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in {"file_search_call", "file_search_tool_call"}:
            continue
        results = item.get("results") or item.get("file_search_call") or []
        if isinstance(results, dict):
            results = results.get("results") or []
        for hit in results or []:
            if not isinstance(hit, dict):
                continue
            sources.append(
                {
                    "content": str(hit.get("text") or hit.get("content") or "")[:500],
                    "score": float(hit.get("score") or hit.get("relevance") or 0.0),
                    "metadata": {
                        "file_id": hit.get("file_id") or hit.get("filename"),
                        "filename": hit.get("filename") or hit.get("file_name"),
                    },
                }
            )
    # include[] sometimes places results under top-level keys
    for key in ("file_search_call.results", "file_search_results"):
        extra = data.get(key)
        if isinstance(extra, list):
            for hit in extra:
                if isinstance(hit, dict):
                    sources.append(
                        {
                            "content": str(hit.get("text") or "")[:500],
                            "score": float(hit.get("score") or 0.0),
                            "metadata": {"file_id": hit.get("file_id")},
                        }
                    )
    return sources
