"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = Field(default="default")
    history: list[ChatMessage] = Field(default_factory=list)
    use_rag: bool = Field(default=True)
    # Reference-free RAGAS-style scores via OGX LLM-as-judge (when sources exist)
    score: bool | None = Field(
        default=None,
        description="Override RAG_EVAL_ON_CHAT; None uses server default",
    )


class SourceChunk(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagMetrics(BaseModel):
    """Reference-free RAGAS-style metrics (no golden docs). Scored via OGX."""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_sufficiency: float | None = None
    avg_retrieval_score: float | None = None
    notes: str = ""
    method: str | None = None
    context_count: int = 0
    eval_latency_ms: float | None = None
    skipped: bool = False
    skip_reason: str | None = None


class ChatResponse(BaseModel):
    answer: str
    model: str | None = None
    latency_ms: float
    trace_id: str
    mlflow_run_id: str | None = None
    session_id: str
    sources: list[SourceChunk] = Field(default_factory=list)
    retrieve_hits: int = 0
    vector_store_id: str | None = None
    guardrails_blocked: bool = False
    guardrails_status: str | None = None
    rag_metrics: RagMetrics | None = None
    eval_mlflow_run_id: str | None = None


class IngestResponse(BaseModel):
    ok: bool
    filename: str
    chunk_count: int
    file_id: str | None = None
    vector_store_id: str | None = None
    collection: str | None = None
    latency_ms: float
    trace_id: str
    mlflow_run_id: str | None = None
    deduped_removed: int = 0


class HealthResponse(BaseModel):
    status: str
    wiring: dict[str, Any]
    flags: dict[str, bool] = Field(default_factory=dict)
    corpus_files: int = 0


class EvaluateRequest(BaseModel):
    """Score a RAG turn without ground-truth / golden documents."""

    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    contexts: list[str] = Field(default_factory=list)
    session_id: str = Field(default="default")
    trace_id: str = Field(default="")
    chat_mlflow_run_id: str = Field(default="")


class EvaluateResponse(BaseModel):
    ok: bool
    faithfulness: float
    answer_relevancy: float
    context_sufficiency: float
    notes: str = ""
    method: str = ""
    context_count: int = 0
    mlflow_run_id: str | None = None
    evalhub: dict[str, Any] = Field(default_factory=dict)
