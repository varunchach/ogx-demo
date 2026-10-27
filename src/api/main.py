"""FastAPI RAG app — OGX Files/Vector Stores + OTel/Tempo + MLflow + EvalHub.

OpenShift AI 3.5: wire only to cluster CRs (OgxServer, MLflow, EvalHub, Tempo
collector, optional standalone NeMo Guardrails). No direct Postgres.
NeMo /v1/guardrail/checks runs before OGX when GUARDRAILS_BASE_URL is set.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    IngestResponse,
    RagMetrics,
    SourceChunk,
)
from src.rag_eval import evaluate_turn
from src.config import Settings, get_settings
from src.evalhub_client import EvalHubClient
from src.guardrails_client import GuardrailsClient
from src.mlflow_logger import MlflowTurnLogger
from src.ogx_client import OgxClient
from src.rag import RagStore
from src.tracing import (
    current_trace_id,
    flush_traces,
    get_tracer,
    instrument_app,
    setup_tracing,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

# MLflow/httpx TLS skip floods logs on customer clusters — hide that noise.
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

_STATIC = Path(__file__).resolve().parents[2] / "static"

_settings: Settings | None = None
_ogx: OgxClient | None = None
_rag: RagStore | None = None
_mlflow: MlflowTurnLogger | None = None
_evalhub: EvalHubClient | None = None
_guardrails: GuardrailsClient | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _settings, _ogx, _rag, _mlflow, _evalhub, _guardrails
    _settings = get_settings()
    setup_tracing(
        _settings.otel_service_name,
        _settings.otel_endpoint,
        _settings.otel_protocol,
    )
    instrument_app(_app)
    _ogx = OgxClient(_settings)
    _rag = RagStore(_ogx)
    _mlflow = MlflowTurnLogger(_settings)
    _evalhub = EvalHubClient(_settings)
    _guardrails = GuardrailsClient(_settings)
    ok, msg = _ogx.ping()
    logger.info("OGX ping ok=%s msg=%s", ok, msg)
    logger.info(
        "OGX embed config model=%s store_name=%s provider=%s llm=%s "
        "(embedding_dimension never sent to OGX)",
        _settings.ogx_embedding_model,
        _settings.ogx_vector_store_name,
        _settings.ogx_vector_provider_id,
        _settings.ogx_model,
    )
    import os as _os

    _gr_env_flag = (_os.environ.get("GUARDRAILS_ENABLED") or "").strip()
    if _settings.guardrails_base_url and not _guardrails.enabled:
        logger.error(
            "NeMo CONFLICT: GUARDRAILS_BASE_URL is set but client disabled — "
            "checks will NOT run"
        )
    if _settings.guardrails_base_url and _gr_env_flag.lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        logger.warning(
            "GUARDRAILS_ENABLED=%r is ignored when BASE_URL is set "
            "(NeMo is ON because URL is present). Clear BASE_URL to disable.",
            _gr_env_flag,
        )
    logger.info(
        "NeMo guardrails startup enabled=%s base_url=%s token_set=%s "
        "GUARDRAILS_ENABLED_env=%r insecure_tls=%s "
        "(chat will log HIT or SKIPPED every turn)",
        _guardrails.enabled,
        _settings.guardrails_base_url or "(unset)",
        bool(_settings.guardrails_api_token),
        _gr_env_flag or "(unset)",
        _settings.guardrails_insecure_tls,
    )
    logger.info("ogx-chatbot ready version=1.4.20 wiring=%s", _settings.wiring_status())
    yield
    flush_traces()


app = FastAPI(title="OGX RAG Chatbot", version="1.4.20", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require() -> tuple[
    Settings, OgxClient, RagStore, MlflowTurnLogger, EvalHubClient, GuardrailsClient
]:
    if not all((_settings, _ogx, _rag, _mlflow, _evalhub, _guardrails)):
        raise HTTPException(status_code=503, detail="App not initialized")
    return _settings, _ogx, _rag, _mlflow, _evalhub, _guardrails  # type: ignore[return-value]


def _flags(settings: Settings) -> dict[str, bool]:
    return {
        "ogx": bool(settings.ogx_base_url),
        "vector_stores": bool(settings.ogx_base_url),
        "mlflow": bool(settings.mlflow_tracking_uri),
        "otel": bool(settings.otel_endpoint),
        "evalhub": bool(settings.evalhub_base_url),
        "embedding_model": bool(settings.ogx_embedding_model),
        "guardrails": bool(settings.guardrails_enabled and settings.guardrails_base_url),
        "rag_eval": bool(settings.rag_eval_on_chat),
    }


@app.get("/readyz")
def readyz() -> dict[str, str]:
    """Lightweight probe — excluded from OTel; does not call OGX."""
    if _settings is None:
        raise HTTPException(status_code=503, detail="starting")
    return {"status": "ready"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings, _, rag, _, _, _ = _require()
    files = 0
    try:
        files = rag.file_count()
    except Exception:  # noqa: BLE001
        logger.exception("file_count failed")
    return HealthResponse(
        status="ok",
        wiring=settings.wiring_status(),
        flags=_flags(settings),
        corpus_files=files,
    )


@app.get("/config")
def config() -> dict[str, Any]:
    settings, ogx, rag, _, evalhub, _ = _require()
    payload = settings.wiring_status()
    payload["flags"] = _flags(settings)
    eh = evalhub.info()
    payload["evalhub_detail"] = eh
    payload["evalhub"] = {
        **eh,
        "base_url": settings.evalhub_base_url or eh.get("base_url"),
        "providers": settings.evalhub_providers or eh.get("providers") or [],
    }
    payload["vector_store_id"] = ogx.vector_store_id
    payload["ogx"] = {
        "model": settings.ogx_model,
        "embedding_model": settings.ogx_embedding_model,
        "vector_store_name": settings.ogx_vector_store_name,
    }
    payload["observability"] = {
        "tempo_hint": (
            "Observe → Traces · service ogx-chatbot-api · "
            "{ name = \"POST /chat\" } or paste trace_id"
        ),
        "mlflow_experiment": settings.mlflow_experiment_name,
        "mlflow_workspace": settings.mlflow_workspace,
    }
    try:
        payload["corpus_files"] = rag.file_count()
    except Exception:  # noqa: BLE001
        payload["corpus_files"] = 0
    return payload


@app.get("/evalhub")
def evalhub_status() -> dict[str, Any]:
    _, _, _, _, evalhub, _ = _require()
    info = evalhub.info()
    info["health"] = evalhub.health()
    return info


@app.get("/debug/guardrails")
def debug_guardrails() -> dict[str, Any]:
    """Show why NeMo gate is HIT vs SKIPPED (no secrets). Live probe optional."""
    import os

    settings, _, _, _, _, guardrails = _require()
    env_keys = {
        k: ("set" if (os.environ.get(k) or "").strip() else "unset")
        for k in (
            "GUARDRAILS_ENABLED",
            "GUARDRAILS_BASE_URL",
            "GUARDRAILS_URL",
            "NEMO_GUARDRAILS_BASE_URL",
            "NEMO_BASE_URL",
            "GUARDRAILS_API_TOKEN",
            "GUARDRAILS_INSECURE_TLS",
            "GUARDRAILS_TIMEOUT_SECONDS",
        )
    }
    probe: dict[str, Any] = {"ran": False}
    if guardrails.enabled:
        result = guardrails.check_user_message(
            "What is the capital of France?",
            history=None,
        )
        probe = {
            "ran": True,
            "status": result.status,
            "blocked": result.blocked,
            "detail": result.detail,
        }
    return {
        "app_version": "1.4.20",
        "client_enabled": guardrails.enabled,
        "settings": {
            "guardrails_enabled": settings.guardrails_enabled,
            "guardrails_base_url": settings.guardrails_base_url or None,
            "api_token_set": bool(settings.guardrails_api_token),
            "insecure_tls": settings.guardrails_insecure_tls,
            "timeout_seconds": settings.guardrails_timeout_seconds,
        },
        "env_keys_present": env_keys,
        "diagnosis": (
            "HIT path active — every /chat must log 'NeMo guardrails HIT'"
            if guardrails.enabled
            else "SKIPPED — GUARDRAILS_BASE_URL not visible to this process "
            "(secret key missing, wrong secret, or pod not restarted after "
            "secret edit). Check env_keys_present."
        ),
        "live_probe": probe,
    }


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate_rag_turn(body: EvaluateRequest) -> EvaluateResponse:
    """Reference-free RAG scores for the last chat turn → MLflow.

    Does not use golden_qa or ground-truth documents. Platform EvalHub
    (batch ragas/garak) remains in the OpenShift AI dashboard.
    """
    settings, ogx, _, mlflow_logger, evalhub, _ = _require()
    if not ogx.configured:
        raise HTTPException(status_code=503, detail="OGX_BASE_URL is not set")
    if not body.contexts:
        raise HTTPException(
            status_code=400,
            detail="No retrieved contexts — ask a document question first "
            "(file_search must return sources).",
        )
    try:
        scores = evaluate_turn(
            ogx,
            question=body.question,
            answer=body.answer,
            contexts=body.contexts,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("evaluate failed")
        raise HTTPException(status_code=502, detail=f"Evaluation failed: {exc}") from exc

    extra_metrics: dict[str, Any] = {
        "vector_store_id": ogx.vector_store_id or "",
        "model": settings.ogx_model or "",
    }
    if scores.get("avg_retrieval_score") is not None:
        extra_metrics["avg_retrieval_score"] = float(scores["avg_retrieval_score"])

    run_id = mlflow_logger.log_evaluation(
        session_id=body.session_id,
        question=body.question,
        faithfulness=float(scores["faithfulness"]),
        answer_relevancy=float(scores["answer_relevancy"]),
        context_sufficiency=float(scores["context_sufficiency"]),
        context_count=int(scores["context_count"]),
        method=str(scores.get("method") or "ogx_llm_judge"),
        trace_id=body.trace_id,
        chat_mlflow_run_id=body.chat_mlflow_run_id,
        notes=str(scores.get("notes") or ""),
        extra=extra_metrics,
    )
    return EvaluateResponse(
        ok=True,
        faithfulness=float(scores["faithfulness"]),
        answer_relevancy=float(scores["answer_relevancy"]),
        context_sufficiency=float(scores["context_sufficiency"]),
        notes=str(scores.get("notes") or ""),
        method=str(scores.get("method") or ""),
        context_count=int(scores["context_count"]),
        mlflow_run_id=run_id,
        evalhub=evalhub.info(),
    )


@app.get("/ogx/models")
def ogx_models() -> dict[str, Any]:
    _, ogx, _, _, _, _ = _require()
    if not ogx.configured:
        raise HTTPException(status_code=503, detail="OGX_BASE_URL is not set")
    try:
        return ogx.list_models()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/debug/rag")
@app.post("/debug/rag")
def debug_rag(q: str = "What is in the uploaded documents?") -> dict[str, Any]:
    """Show file_search hits, store files, and a clear diagnosis string."""
    _, ogx, _, _, _, _ = _require()
    if not ogx.configured:
        raise HTTPException(status_code=503, detail="OGX_BASE_URL is not set")
    try:
        result = ogx.debug_rag(q)
        logger.info(
            "debug/rag diagnosis=%s retrieve_hits=%s files=%s",
            result.get("diagnosis"),
            result.get("retrieve_hits"),
            result.get("files_count"),
        )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("debug/rag failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)) -> IngestResponse:
    """Ingest: local chunk → OGX embeddings (batches of 16) → vector_io.insert.

    Sync ``def`` so embed/insert HTTP work runs in the threadpool and probes
    stay healthy. Matches the validated notebook path (not file-attach auto-embed).
    """
    settings, ogx, rag, mlflow_logger, _, _ = _require()
    if not ogx.configured:
        raise HTTPException(status_code=503, detail="OGX_BASE_URL is not set")

    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    filename = file.filename or "upload.txt"
    logger.info(
        "ingest accepted filename=%s bytes=%s (local chunk + OGX embed batches)",
        filename,
        len(data),
    )

    tracer = get_tracer()
    with tracer.start_as_current_span("ingest") as span:
        span.set_attribute("ingest.filename", filename)
        span.set_attribute("ingest.bytes", len(data))
        started = time.perf_counter()
        try:
            result = rag.ingest_bytes(filename, data)
        except Exception as exc:  # noqa: BLE001
            span.record_exception(exc)
            raise HTTPException(status_code=502, detail=f"OGX ingest failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        trace_id = current_trace_id()
        span.set_attribute("chunk.count", result.get("chunk_count") or 0)
        span.set_attribute("latency_ms", latency_ms)

        run_id = mlflow_logger.log_ingest(
            filename=result["filename"],
            chunk_count=int(result.get("chunk_count") or 0),
            latency_ms=latency_ms,
            trace_id=trace_id,
            embedding_model=settings.ogx_embedding_model or "ogx-default",
            extra={
                "vector_store_id": result.get("vector_store_id") or "",
                "file_id": result.get("file_id") or "",
            },
        )

    flush_traces()
    return IngestResponse(
        ok=True,
        filename=result["filename"],
        chunk_count=int(result.get("chunk_count") or 0),
        file_id=result.get("file_id"),
        vector_store_id=result.get("vector_store_id"),
        collection=result.get("collection"),
        latency_ms=round(latency_ms, 2),
        trace_id=trace_id,
        mlflow_run_id=run_id,
        deduped_removed=int(result.get("deduped_removed") or 0),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    settings, ogx, rag, mlflow_logger, _, guardrails = _require()
    if not ogx.configured:
        raise HTTPException(status_code=503, detail="OGX_BASE_URL is not set")

    tracer = get_tracer()
    history = [{"role": m.role, "content": m.content} for m in body.history]
    sources: list[SourceChunk] = []
    retrieve_hits = 0

    with tracer.start_as_current_span("chat.turn") as span:
        span.set_attribute("gen_ai.system", "ogx")
        span.set_attribute("gen_ai.request.model", settings.ogx_model or "")
        span.set_attribute("session.id", body.session_id)
        span.set_attribute("rag.enabled", body.use_rag)
        span.set_attribute("user.message", body.message[:500])
        span.set_attribute("guardrails.enabled", guardrails.enabled)

        started = time.perf_counter()

        with tracer.start_as_current_span("guardrails.checks") as gr_span:
            gr = guardrails.check_user_message(body.message, history=history)
            gr_span.set_attribute("guardrails.enabled", guardrails.enabled)
            gr_span.set_attribute("guardrails.status", gr.status)
            gr_span.set_attribute("guardrails.blocked", gr.blocked)
            gr_span.set_attribute("guardrails.detail", (gr.detail or "")[:200])

        logger.info(
            "chat guardrails gate enabled=%s status=%s blocked=%s detail=%s "
            "session=%s (must appear before rag_chat / OGX)",
            guardrails.enabled,
            gr.status,
            gr.blocked,
            gr.detail,
            body.session_id,
        )

        if gr.blocked:
            latency_ms = (time.perf_counter() - started) * 1000.0
            trace_id = current_trace_id()
            answer = (
                "Your message was blocked by safety guardrails and was not "
                f"sent to the model. ({gr.detail})"
            )
            span.set_attribute("guardrails.blocked", True)
            span.set_attribute("guardrails.status", gr.status)
            span.set_attribute("assistant.message", answer[:500])
            span.set_attribute("latency_ms", latency_ms)
            span.set_attribute("http.route", "/chat")
            logger.warning(
                "chat blocked by guardrails session=%s status=%s detail=%s "
                "trace_id=%s",
                body.session_id,
                gr.status,
                gr.detail,
                trace_id,
            )
            run_id = mlflow_logger.log_turn(
                user_message=body.message,
                assistant_message=answer,
                model=settings.ogx_model,
                latency_ms=latency_ms,
                trace_id=trace_id,
                session_id=body.session_id,
                extra={
                    "retrieve_hits": 0,
                    "use_rag": str(body.use_rag),
                    "vector_store_id": ogx.vector_store_id or "",
                    "guardrails_blocked": "true",
                    "guardrails_status": gr.status,
                },
            )
            flush_traces()
            return ChatResponse(
                answer=answer,
                model=settings.ogx_model or None,
                latency_ms=round(latency_ms, 2),
                trace_id=trace_id,
                mlflow_run_id=run_id,
                session_id=body.session_id,
                sources=[],
                retrieve_hits=0,
                vector_store_id=ogx.vector_store_id,
                guardrails_blocked=True,
                guardrails_status=gr.status,
            )

        try:
            if body.use_rag:
                with tracer.start_as_current_span("rag.generate") as rag_span:
                    rag_span.set_attribute("gen_ai.operation.name", "chat")
                    answer, raw_sources = rag.rag_chat(body.message, history=history)
                sources = [
                    SourceChunk(
                        content=s.get("content") or "",
                        score=float(s.get("score") or 0.0),
                        metadata=s.get("metadata") or {},
                    )
                    for s in raw_sources
                ]
                retrieve_hits = len(sources)
            else:
                with tracer.start_as_current_span("llm.chat"):
                    answer = ogx.chat(body.message, history=history)
        except Exception as exc:  # noqa: BLE001
            span.record_exception(exc)
            span.set_attribute("error", True)
            raise HTTPException(status_code=502, detail=f"OGX/RAG failed: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000.0
        trace_id = current_trace_id()
        span.set_attribute("guardrails.blocked", False)
        span.set_attribute("guardrails.status", gr.status)
        span.set_attribute("assistant.message", answer[:500])
        span.set_attribute("latency_ms", latency_ms)
        span.set_attribute("retrieve.hits", retrieve_hits)
        span.set_attribute("http.route", "/chat")
        logger.info(
            "chat turn session=%s use_rag=%s retrieve_hits=%s latency_ms=%.1f "
            "trace_id=%s store=%s guardrails=%s answer_preview=%r",
            body.session_id,
            body.use_rag,
            retrieve_hits,
            latency_ms,
            trace_id,
            ogx.vector_store_id,
            gr.status,
            (answer or "")[:200],
        )
        if body.use_rag and retrieve_hits == 0:
            logger.warning(
                "chat turn had use_rag=true but retrieve_hits=0 — check OGX "
                "file_search / embeddings (see ogx_client logs above)"
            )

        want_score = (
            settings.rag_eval_on_chat if body.score is None else bool(body.score)
        )
        rag_metrics: RagMetrics | None = None
        eval_run_id: str | None = None
        if want_score and body.use_rag:
            contexts = [s.content for s in sources if (s.content or "").strip()]
            if settings.rag_eval_require_contexts and not contexts:
                rag_metrics = RagMetrics(
                    skipped=True,
                    skip_reason="no retrieved contexts",
                    context_count=0,
                    method="ogx_llm_judge_reference_free",
                )
            else:
                with tracer.start_as_current_span("rag.eval") as eval_span:
                    eval_started = time.perf_counter()
                    try:
                        scores = evaluate_turn(
                            ogx,
                            question=body.message,
                            answer=answer,
                            contexts=contexts,
                            retrieval_scores=[s.score for s in sources],
                        )
                        eval_ms = (time.perf_counter() - eval_started) * 1000.0
                        rag_metrics = RagMetrics(
                            faithfulness=float(scores["faithfulness"]),
                            answer_relevancy=float(scores["answer_relevancy"]),
                            context_sufficiency=float(scores["context_sufficiency"]),
                            avg_retrieval_score=(
                                float(scores["avg_retrieval_score"])
                                if scores.get("avg_retrieval_score") is not None
                                else None
                            ),
                            notes=str(scores.get("notes") or ""),
                            method=str(scores.get("method") or ""),
                            context_count=int(scores["context_count"]),
                            eval_latency_ms=round(eval_ms, 2),
                            skipped=False,
                        )
                        eval_span.set_attribute(
                            "rag.faithfulness", rag_metrics.faithfulness or 0.0
                        )
                        eval_span.set_attribute(
                            "rag.answer_relevancy",
                            rag_metrics.answer_relevancy or 0.0,
                        )
                        eval_span.set_attribute(
                            "rag.context_sufficiency",
                            rag_metrics.context_sufficiency or 0.0,
                        )
                        eval_span.set_attribute("rag.eval_latency_ms", eval_ms)
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("inline rag_eval failed; chat still returned")
                        eval_span.record_exception(exc)
                        rag_metrics = RagMetrics(
                            skipped=True,
                            skip_reason=f"eval failed: {exc}",
                            context_count=len(contexts),
                            method="ogx_llm_judge_reference_free",
                        )

        turn_extra: dict[str, Any] = {
            "retrieve_hits": retrieve_hits,
            "use_rag": str(body.use_rag),
            "vector_store_id": ogx.vector_store_id or "",
            "guardrails_enabled": str(guardrails.enabled),
            "guardrails_blocked": "false",
            "guardrails_status": gr.status,
            "guardrails_detail": (gr.detail or "")[:250],
        }
        mlflow_rag: dict[str, Any] | None = None
        if rag_metrics and not rag_metrics.skipped:
            mlflow_rag = {
                "faithfulness": float(rag_metrics.faithfulness or 0.0),
                "answer_relevancy": float(rag_metrics.answer_relevancy or 0.0),
                "context_sufficiency": float(rag_metrics.context_sufficiency or 0.0),
                "context_count": int(rag_metrics.context_count),
                "retrieve_hits": retrieve_hits,
                "method": str(rag_metrics.method or ""),
                "notes": rag_metrics.notes or "",
            }
            if rag_metrics.avg_retrieval_score is not None:
                mlflow_rag["avg_retrieval_score"] = float(
                    rag_metrics.avg_retrieval_score
                )
            if rag_metrics.eval_latency_ms is not None:
                mlflow_rag["eval_latency_ms"] = float(rag_metrics.eval_latency_ms)

        # Chat run first so metrics land on the turn the customer opens in MLflow.
        run_id = mlflow_logger.log_turn(
            user_message=body.message,
            assistant_message=answer,
            model=settings.ogx_model,
            latency_ms=latency_ms,
            trace_id=trace_id,
            session_id=body.session_id,
            extra=turn_extra,
            rag_metrics=mlflow_rag,
        )

        # Linked sibling eval run (filterable by kind=rag_evaluation).
        if mlflow_rag is not None:
            eval_run_id = mlflow_logger.log_evaluation(
                session_id=body.session_id,
                question=body.message,
                faithfulness=float(mlflow_rag["faithfulness"]),
                answer_relevancy=float(mlflow_rag["answer_relevancy"]),
                context_sufficiency=float(mlflow_rag["context_sufficiency"]),
                context_count=int(mlflow_rag["context_count"]),
                method=str(mlflow_rag.get("method") or "ogx_llm_judge"),
                trace_id=trace_id,
                chat_mlflow_run_id=run_id or "",
                notes=str(mlflow_rag.get("notes") or ""),
                extra={
                    "vector_store_id": ogx.vector_store_id or "",
                    "model": settings.ogx_model or "",
                    **(
                        {"eval_latency_ms": float(mlflow_rag["eval_latency_ms"])}
                        if mlflow_rag.get("eval_latency_ms") is not None
                        else {}
                    ),
                    **(
                        {
                            "avg_retrieval_score": float(
                                mlflow_rag["avg_retrieval_score"]
                            )
                        }
                        if mlflow_rag.get("avg_retrieval_score") is not None
                        else {}
                    ),
                },
            )

    flush_traces()
    return ChatResponse(
        answer=answer,
        model=settings.ogx_model or None,
        latency_ms=round(latency_ms, 2),
        trace_id=trace_id,
        mlflow_run_id=run_id,
        session_id=body.session_id,
        sources=sources,
        retrieve_hits=retrieve_hits,
        vector_store_id=ogx.vector_store_id,
        guardrails_blocked=False,
        guardrails_status=gr.status,
        rag_metrics=rag_metrics,
        eval_mlflow_run_id=eval_run_id,
    )


@app.get("/")
def index() -> FileResponse:
    index_path = _STATIC / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_path)


if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")
