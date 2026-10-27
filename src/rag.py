"""RAG orchestration via OgxServer APIs (no direct Postgres).

Ingest: local chunk → OGX /v1/embeddings (batches) → vector_io.insert.
Retrieve: vector_io.query + chat/completions (file_search fallback).
"""

from __future__ import annotations

from typing import Any

from src.ogx_client import OgxClient
from src.tracing import get_tracer


class RagStore:
    """Thin façade over OgxClient vector-store RAG."""

    def __init__(self, ogx: OgxClient) -> None:
        self._ogx = ogx

    @property
    def configured(self) -> bool:
        return self._ogx.configured

    def ping(self) -> tuple[bool, str]:
        return self._ogx.ping()

    def file_count(self) -> int:
        return self._ogx.file_count()

    def ingest_bytes(self, filename: str, data: bytes) -> dict[str, Any]:
        tracer = get_tracer()
        with tracer.start_as_current_span("ingest.document") as span:
            span.set_attribute("ingest.filename", filename)
            span.set_attribute("ingest.bytes", len(data))
            result = self._ogx.ingest_bytes(filename, data)
            span.set_attribute("vector_store.id", result.get("vector_store_id") or "")
            span.set_attribute("file.id", result.get("file_id") or "")
            span.set_attribute("chunk.count", int(result.get("chunk_count") or 0))
            return result

    def rag_chat(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        tracer = get_tracer()
        with tracer.start_as_current_span("rag.file_search") as span:
            answer, sources = self._ogx.rag_chat(message, history=history)
            span.set_attribute("retrieve.hits", len(sources))
            span.set_attribute("answer.chars", len(answer))
            return answer, sources
