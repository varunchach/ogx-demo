"""Best-effort MLflow logging against the cluster MLflow operator URI."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.config import Settings

logger = logging.getLogger(__name__)


def _inject_workspace_header() -> None:
    """RHOAI MLflow needs x-mlflow-workspace + Bearer SA token on every call."""
    try:
        from mlflow.tracking.request_header.abstract_request_header_provider import (
            RequestHeaderProvider,
        )
        from mlflow.tracking.request_header.registry import _request_header_provider_registry

        class _WorkspaceProvider(RequestHeaderProvider):
            def in_context(self) -> bool:
                return True

            def request_headers(self) -> dict[str, str]:
                headers = {
                    "x-mlflow-workspace": os.environ.get("MLFLOW_WORKSPACE", ""),
                }
                token = os.environ.get("MLFLOW_TRACKING_TOKEN", "")
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                return headers

        _request_header_provider_registry.register(_WorkspaceProvider)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not inject MLflow workspace header: %s", exc)


class MlflowTurnLogger:
    """Best-effort MLflow logging; never fails the chat/ingest path."""

    def __init__(self, settings: Settings) -> None:
        self._uri = settings.mlflow_tracking_uri
        self._experiment = settings.mlflow_experiment_name
        self._workspace = settings.mlflow_workspace
        self._insecure = settings.mlflow_insecure_tls
        self._ready = False
        if self._uri:
            self._init_mlflow()

    @property
    def configured(self) -> bool:
        return bool(self._uri)

    def _init_mlflow(self) -> None:
        try:
            import mlflow

            if self._insecure:
                os.environ.setdefault("MLFLOW_TRACKING_INSECURE_TLS", "true")
            mlflow.set_tracking_uri(self._uri)

            token_path = Path(
                os.environ.get(
                    "MLFLOW_TRACKING_TOKEN_PATH",
                    "/var/run/secrets/kubernetes.io/serviceaccount/token",
                )
            )
            if not os.environ.get("MLFLOW_TRACKING_TOKEN") and token_path.is_file():
                os.environ["MLFLOW_TRACKING_TOKEN"] = token_path.read_text(
                    encoding="utf-8"
                ).strip()
            # Prefer explicit Bearer header injection (below) over
            # MLFLOW_TRACKING_AUTH=kubernetes-namespaced, which requires
            # the optional mlflow[kubernetes] extra not present in the image.

            # Workspaces map 1:1 to namespaces with opendatahub.io/dashboard=true.
            candidates: list[str | None] = []
            if self._workspace:
                candidates.append(self._workspace)
            for name in ("models-as-a-service", "credit-decisioning", "external-models"):
                if name not in candidates:
                    candidates.append(name)
            candidates.append(None)

            _inject_workspace_header()

            last_err: Exception | None = None
            for workspace in candidates:
                try:
                    if workspace:
                        os.environ["MLFLOW_WORKSPACE"] = workspace
                    else:
                        os.environ.pop("MLFLOW_WORKSPACE", None)
                    mlflow.set_experiment(self._experiment)
                    self._ready = True
                    logger.info(
                        "MLflow -> %s experiment=%s workspace=%s",
                        self._uri,
                        self._experiment,
                        workspace or "(default)",
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    logger.warning(
                        "MLflow workspace=%s failed (%s); trying next",
                        workspace,
                        exc,
                    )
            raise RuntimeError(f"No usable MLflow workspace: {last_err}")
        except Exception:  # noqa: BLE001 — demo must stay up if MLflow is down
            logger.exception("MLflow init failed; turn logging disabled")
            self._ready = False

    def log_evaluation(
        self,
        *,
        session_id: str,
        question: str,
        faithfulness: float,
        answer_relevancy: float,
        context_sufficiency: float,
        context_count: int,
        method: str,
        trace_id: str = "",
        chat_mlflow_run_id: str = "",
        notes: str = "",
        extra: dict[str, Any] | None = None,
    ) -> str | None:
        """Log reference-free RAG scores alongside chat runs."""
        if not self._ready:
            return None
        try:
            import mlflow

            with mlflow.start_run(run_name=f"eval-{session_id[:8]}") as run:
                mlflow.log_param("kind", "rag_evaluation")
                mlflow.log_param("session_id", session_id)
                mlflow.log_param("method", method[:120])
                mlflow.log_param("question", question[:500])
                mlflow.log_param("trace_id", trace_id or "")
                mlflow.log_param("chat_mlflow_run_id", chat_mlflow_run_id or "")
                if notes:
                    mlflow.log_param("notes", notes[:500])
                mlflow.log_metric("faithfulness", float(faithfulness))
                mlflow.log_metric("answer_relevancy", float(answer_relevancy))
                mlflow.log_metric("context_sufficiency", float(context_sufficiency))
                mlflow.log_metric("context_count", float(context_count))
                if extra:
                    for key, value in extra.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, float(value))
                        else:
                            mlflow.log_param(key, str(value)[:250])
                return run.info.run_id
        except Exception:  # noqa: BLE001
            logger.exception("MLflow log_evaluation failed")
            return None

    def log_turn(
        self,
        *,
        user_message: str,
        assistant_message: str,
        model: str,
        latency_ms: float,
        trace_id: str,
        session_id: str,
        extra: dict[str, Any] | None = None,
        rag_metrics: dict[str, Any] | None = None,
    ) -> str | None:
        """Log a chat turn; optional rag_metrics become first-class MLflow metrics."""
        if not self._ready:
            return None
        try:
            import mlflow

            with mlflow.start_run(run_name=f"chat-{session_id[:8]}") as run:
                mlflow.log_param("kind", "chat")
                mlflow.log_param("model", model or "unknown")
                mlflow.log_param("session_id", session_id)
                mlflow.log_param("trace_id", trace_id or "")
                mlflow.log_param("user_message", user_message[:500])
                mlflow.log_param("assistant_message", assistant_message[:500])
                mlflow.log_metric("latency_ms", latency_ms)
                if rag_metrics:
                    method = str(rag_metrics.get("method") or "")
                    if method:
                        mlflow.log_param("rag_eval_method", method[:120])
                    notes = str(rag_metrics.get("notes") or "")
                    if notes:
                        mlflow.log_param("rag_eval_notes", notes[:500])
                    for key in (
                        "faithfulness",
                        "answer_relevancy",
                        "context_sufficiency",
                        "avg_retrieval_score",
                        "context_count",
                        "eval_latency_ms",
                        "retrieve_hits",
                    ):
                        if rag_metrics.get(key) is None:
                            continue
                        try:
                            mlflow.log_metric(key, float(rag_metrics[key]))
                        except (TypeError, ValueError):
                            continue
                    logger.info(
                        "MLflow chat run=%s logged rag metrics faithfulness=%s "
                        "relevancy=%s sufficiency=%s",
                        run.info.run_id,
                        rag_metrics.get("faithfulness"),
                        rag_metrics.get("answer_relevancy"),
                        rag_metrics.get("context_sufficiency"),
                    )
                if extra:
                    skip_keys = (
                        {
                            "faithfulness",
                            "answer_relevancy",
                            "context_sufficiency",
                            "avg_retrieval_score",
                            "context_count",
                            "eval_latency_ms",
                            "retrieve_hits",
                        }
                        if rag_metrics
                        else set()
                    )
                    for key, value in extra.items():
                        if key in skip_keys:
                            continue
                        if isinstance(value, (int, float)) and not isinstance(
                            value, bool
                        ):
                            mlflow.log_metric(key, float(value))
                        else:
                            mlflow.log_param(key, str(value)[:250])
                return run.info.run_id
        except Exception:  # noqa: BLE001
            logger.exception("MLflow log_turn failed")
            return None

    def log_ingest(
        self,
        *,
        filename: str,
        chunk_count: int,
        latency_ms: float,
        trace_id: str,
        embedding_model: str,
        extra: dict[str, Any] | None = None,
    ) -> str | None:
        if not self._ready:
            return None
        try:
            import mlflow

            with mlflow.start_run(run_name=f"ingest-{filename[:24]}") as run:
                mlflow.log_param("kind", "ingest")
                mlflow.log_param("filename", filename[:250])
                mlflow.log_param("embedding_model", embedding_model or "unknown")
                mlflow.log_param("trace_id", trace_id or "")
                mlflow.log_metric("chunk_count", float(chunk_count))
                mlflow.log_metric("latency_ms", latency_ms)
                if extra:
                    for key, value in extra.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(key, float(value))
                        else:
                            mlflow.log_param(key, str(value)[:250])
                return run.info.run_id
        except Exception:  # noqa: BLE001
            logger.exception("MLflow log_ingest failed")
            return None
