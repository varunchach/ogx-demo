"""Load settings from env vars (preferred) with YAML defaults.

Platform-only: OGX, MLflow, EvalHub, Tempo/collector URLs come from
cluster-installed CRs. This module never hardcodes hosts.
No direct Postgres — RAG goes through OGX Files / Vector Stores APIs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def _load_yaml() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    # Strip accidental quotes pasted into Secret stringData / console.
    return value.strip().strip('"').strip("'")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw.strip())


@dataclass(frozen=True)
class Settings:
    app_name: str
    ogx_base_url: str
    ogx_api_key: str
    ogx_model: str
    ogx_embedding_model: str
    ogx_vector_store_id: str
    ogx_vector_store_name: str
    ogx_vector_provider_id: str
    ogx_timeout_seconds: float
    ogx_embedding_dimension: int
    ogx_embedding_tokenizer: str
    ogx_embed_batch_size: int
    ogx_ingest_dedupe: bool
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    otel_endpoint: str
    otel_protocol: str
    otel_service_name: str
    mlflow_tracking_uri: str
    mlflow_insecure_tls: bool
    mlflow_experiment_name: str
    mlflow_workspace: str
    evalhub_base_url: str
    evalhub_api_token: str
    rag_eval_on_chat: bool
    rag_eval_require_contexts: bool
    guardrails_enabled: bool
    guardrails_base_url: str
    guardrails_api_token: str
    guardrails_timeout_seconds: float
    guardrails_insecure_tls: bool
    # Must stay after required fields (dataclass default-arg rule)
    evalhub_providers: list[str] = field(default_factory=list)

    def wiring_status(self) -> dict[str, Any]:
        """Non-secret readiness for /health, /config, and the UI."""
        return {
            "ogx": {
                "configured": bool(self.ogx_base_url),
                "base_url": self.ogx_base_url or None,
                "model": self.ogx_model or None,
                "embedding_model": self.ogx_embedding_model or None,
                "vector_store_id": self.ogx_vector_store_id or None,
                "vector_store_name": self.ogx_vector_store_name,
                "vector_provider_id": self.ogx_vector_provider_id,
                "embedding_dimension": self.ogx_embedding_dimension or None,
                "embedding_tokenizer": self.ogx_embedding_tokenizer or None,
                "embed_batch_size": self.ogx_embed_batch_size,
                "api_key_set": bool(self.ogx_api_key),
                "ingest_dedupe": self.ogx_ingest_dedupe,
            },
            "rag": {
                "via": "local_chunk_ogx_embed_vector_io",
                "chunk_size_tokens": self.chunk_size_tokens,
                "chunk_overlap_tokens": self.chunk_overlap_tokens,
                "embed_batch_size": self.ogx_embed_batch_size,
                "ingest_dedupe_by_filename": self.ogx_ingest_dedupe,
                "note": (
                    "App chunks locally; OGX /v1/embeddings in batches; "
                    "vector_io.insert — matches validated notebook path"
                ),
            },
            "guardrails": {
                "enabled": self.guardrails_enabled and bool(self.guardrails_base_url),
                "base_url": self.guardrails_base_url or None,
                "api_token_set": bool(self.guardrails_api_token),
                "api_token_optional": True,
                "endpoint": "/v1/guardrail/checks",
                "note": "Standalone NeMo — checks before OGX; GUARDRAILS_API_TOKEN optional",
            },
            "otel": {
                "configured": bool(self.otel_endpoint),
                "endpoint": self.otel_endpoint or None,
                "protocol": self.otel_protocol,
                "service_name": self.otel_service_name,
            },
            "mlflow": {
                "configured": bool(self.mlflow_tracking_uri),
                "tracking_uri": self.mlflow_tracking_uri or None,
                "experiment_name": self.mlflow_experiment_name,
                "workspace": self.mlflow_workspace or None,
                "insecure_tls": self.mlflow_insecure_tls,
            },
            "evalhub": {
                "configured": bool(self.evalhub_base_url),
                "base_url": self.evalhub_base_url or None,
                "api_token_set": bool(self.evalhub_api_token),
                "providers": self.evalhub_providers,
                "note": "Batch/dashboard only — live RAGAS-style scores use OGX LLM-as-judge",
            },
            "rag_eval": {
                "on_chat": self.rag_eval_on_chat,
                "require_contexts": self.rag_eval_require_contexts,
                "method": "ogx_llm_judge_reference_free",
                "metrics": [
                    "faithfulness",
                    "answer_relevancy",
                    "context_sufficiency",
                    "avg_retrieval_score",
                ],
            },
        }


def get_settings() -> Settings:
    raw = _load_yaml()
    ogx = raw.get("ogx") or {}
    chunk = raw.get("chunking") or {}
    otel = raw.get("otel") or {}
    mlflow = raw.get("mlflow") or {}
    evalhub = raw.get("evalhub") or {}
    rag_eval = raw.get("rag_eval") or {}
    guardrails = raw.get("guardrails") or {}
    app = raw.get("app") or {}

    providers_raw = _env(
        "EVALHUB_PROVIDERS",
        str(evalhub.get("providers") or "ragas,lm-evaluation-harness,garak"),
    )
    providers = [p.strip() for p in providers_raw.split(",") if p.strip()]

    gr_url = _env(
        "GUARDRAILS_BASE_URL", str(guardrails.get("base_url") or "")
    ).rstrip("/")
    # Also accept common typo / alternate secret key names.
    if not gr_url:
        gr_url = (
            _env("GUARDRAILS_URL", "")
            or _env("NEMO_GUARDRAILS_BASE_URL", "")
            or _env("NEMO_BASE_URL", "")
        ).rstrip("/")
    # Source of truth = URL. Lab/customer secrets often leave GUARDRAILS_ENABLED=false
    # after pasting the Route; that used to silently skip NeMo despite a valid URL.
    # To disable NeMo: clear GUARDRAILS_BASE_URL (do not rely on ENABLED=false alone).
    gr_enabled = bool(gr_url)

    return Settings(
        app_name=_env("APP_NAME", str(app.get("name") or "ogx-chatbot")),
        ogx_base_url=_env("OGX_BASE_URL", str(ogx.get("base_url") or "")),
        ogx_api_key=_env("OGX_API_KEY", str(ogx.get("api_key") or "")),
        ogx_model=_env("OGX_MODEL", str(ogx.get("model") or "")),
        ogx_embedding_model=_env(
            "OGX_EMBEDDING_MODEL", str(ogx.get("embedding_model") or "")
        ),
        ogx_vector_store_id=_env(
            "OGX_VECTOR_STORE_ID", str(ogx.get("vector_store_id") or "")
        ),
        ogx_vector_store_name=_env(
            "OGX_VECTOR_STORE_NAME",
            str(ogx.get("vector_store_name") or "ogx-rag-demo"),
        ),
        ogx_vector_provider_id=_env(
            "OGX_VECTOR_PROVIDER_ID",
            str(ogx.get("vector_provider_id") or "pgvector"),
        ),
        ogx_timeout_seconds=float(
            _env("OGX_TIMEOUT_SECONDS", str(ogx.get("timeout_seconds") or 180))
        ),
        ogx_embedding_dimension=_env_int(
            "OGX_EMBEDDING_DIMENSION",
            int(ogx.get("embedding_dimension") or 1024),
        ),
        ogx_embedding_tokenizer=_env(
            "OGX_EMBEDDING_TOKENIZER",
            str(ogx.get("embedding_tokenizer") or ""),
        ),
        ogx_embed_batch_size=_env_int(
            "OGX_EMBED_BATCH_SIZE",
            int(ogx.get("embed_batch_size") or 16),
        ),
        ogx_ingest_dedupe=_env_bool(
            "OGX_INGEST_DEDUPE",
            bool(ogx.get("ingest_dedupe", True)),
        ),
        chunk_size_tokens=_env_int(
            "CHUNK_SIZE_TOKENS", int(chunk.get("chunk_size_tokens") or 384)
        ),
        chunk_overlap_tokens=_env_int(
            "CHUNK_OVERLAP_TOKENS", int(chunk.get("chunk_overlap_tokens") or 64)
        ),
        otel_endpoint=_env(
            "OTEL_EXPORTER_OTLP_ENDPOINT", str(otel.get("endpoint") or "")
        ),
        otel_protocol=_env(
            "OTEL_EXPORTER_OTLP_PROTOCOL",
            str(otel.get("protocol") or "http/protobuf"),
        ),
        otel_service_name=_env(
            "OTEL_SERVICE_NAME",
            str(otel.get("service_name") or app.get("service_name") or "ogx-chatbot-api"),
        ),
        mlflow_tracking_uri=_env(
            "MLFLOW_TRACKING_URI", str(mlflow.get("tracking_uri") or "")
        ),
        mlflow_insecure_tls=_env_bool(
            "MLFLOW_TRACKING_INSECURE_TLS",
            bool(mlflow.get("insecure_tls", True)),
        ),
        mlflow_experiment_name=_env(
            "MLFLOW_EXPERIMENT_NAME",
            str(mlflow.get("experiment_name") or "ogx-chatbot"),
        ),
        mlflow_workspace=_env(
            "MLFLOW_WORKSPACE", str(mlflow.get("workspace") or "")
        ),
        evalhub_base_url=_env(
            "EVALHUB_BASE_URL", str(evalhub.get("base_url") or "")
        ),
        evalhub_api_token=_env(
            "EVALHUB_API_TOKEN", str(evalhub.get("api_token") or "")
        ),
        evalhub_providers=providers,
        rag_eval_on_chat=_env_bool(
            "RAG_EVAL_ON_CHAT", bool(rag_eval.get("on_chat", True))
        ),
        rag_eval_require_contexts=_env_bool(
            "RAG_EVAL_REQUIRE_CONTEXTS",
            bool(rag_eval.get("require_contexts", True)),
        ),
        guardrails_enabled=gr_enabled,
        guardrails_base_url=gr_url,
        guardrails_api_token=_env(
            "GUARDRAILS_API_TOKEN", str(guardrails.get("api_token") or "")
        ),
        guardrails_timeout_seconds=float(
            _env(
                "GUARDRAILS_TIMEOUT_SECONDS",
                str(guardrails.get("timeout_seconds") or 60),
            )
        ),
        guardrails_insecure_tls=_env_bool(
            "GUARDRAILS_INSECURE_TLS",
            bool(guardrails.get("insecure_tls", True)),
        ),
    )
