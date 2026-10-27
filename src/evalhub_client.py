"""EvalHub wiring — status, providers from env, optional health check.

EvalHub does not ingest live chat turns. This client exposes configured
providers and a deep-link / health probe so the UI can showcase evaluation.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from src.config import Settings

logger = logging.getLogger(__name__)


class EvalHubClient:
    def __init__(self, settings: Settings) -> None:
        self._base = settings.evalhub_base_url.rstrip("/")
        self._token = settings.evalhub_api_token
        self.providers = list(settings.evalhub_providers)

    @property
    def configured(self) -> bool:
        return bool(self._base)

    def info(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "base_url": self._base or None,
            "providers": self.providers,
            "in_app_evaluation": True,
            "dashboard_hint": (
                "Platform EvalHub (ragas / lm-eval / garak) is batch-only in the "
                "OpenShift AI dashboard (Evaluations). It cannot score live chat "
                "turns. Live reference-free RAGAS-style metrics "
                "(faithfulness / answer_relevancy / context_sufficiency) are "
                "computed in-app via OGX LLM-as-judge and logged to MLflow."
            ),
            "how_to_open_platform": (
                "OpenShift AI dashboard → Evaluations (while logged in). "
                "Do not use the raw EvalHub Route from this app."
            ),
        }

    def health(self) -> dict[str, Any]:
        if not self.configured:
            return {"ok": False, "detail": "EVALHUB_BASE_URL is not set"}
        headers: dict[str, str] = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            with httpx.Client(timeout=15.0, verify=False) as client:
                # Best-effort probe; EvalHub route layouts vary by version.
                for path in ("/health", "/api/health", "/", "/q/health"):
                    resp = client.get(f"{self._base}{path}", headers=headers)
                    if resp.status_code < 500:
                        return {
                            "ok": resp.status_code < 400,
                            "status_code": resp.status_code,
                            "path": path,
                            "base_url": self._base,
                            "providers": self.providers,
                        }
            return {
                "ok": False,
                "detail": "No healthy EvalHub path responded",
                "base_url": self._base,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("EvalHub health probe failed: %s", exc)
            return {
                "ok": False,
                "detail": str(exc),
                "base_url": self._base,
                "providers": self.providers,
            }
