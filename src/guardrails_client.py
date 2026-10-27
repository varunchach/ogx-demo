"""Standalone NeMo Guardrails client — /v1/guardrail/checks only (no LLM).

Official RHOAI path: call checks before OGX; do not replace MaaS with NeMo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

from src.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GuardrailsResult:
    blocked: bool
    status: str
    detail: str
    raw: dict[str, Any]


class GuardrailsClient:
    """POST {base}/v1/guardrail/checks when GUARDRAILS_BASE_URL is set.

    GUARDRAILS_API_TOKEN is optional — Authorization is sent only when set.
    """

    def __init__(self, settings: Settings) -> None:
        self._base = (settings.guardrails_base_url or "").rstrip("/")
        self._token = (settings.guardrails_api_token or "").strip()
        self._timeout = settings.guardrails_timeout_seconds
        self._verify = not settings.guardrails_insecure_tls
        self._enabled = bool(self._base) and settings.guardrails_enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def check_user_message(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
    ) -> GuardrailsResult:
        """Validate user text; skip (allow) when guardrails are not configured."""
        if not self._enabled:
            if not self._base:
                detail = "skipped: GUARDRAILS_BASE_URL unset"
            else:
                detail = "skipped: GUARDRAILS_ENABLED=false"
            logger.warning(
                "NeMo guardrails SKIPPED detail=%s base_url=%r token_set=%s",
                detail,
                self._base or "",
                bool(self._token),
            )
            return GuardrailsResult(
                blocked=False,
                status="skipped",
                detail=detail,
                raw={},
            )

        # Input rails: current user turn ONLY. Prior chat history must not be
        # sent — a earlier sensitive user message would re-trigger blocks.
        del history
        messages = [{"role": "user", "content": message}]

        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        url = f"{self._base}/v1/guardrail/checks"
        payload = {"model": "test", "messages": messages}
        logger.info(
            "NeMo guardrails HIT calling url=%s token_set=%s msg_preview=%r "
            "(current turn only, no history)",
            url,
            bool(self._token),
            message[:120],
        )
        try:
            with httpx.Client(timeout=self._timeout, verify=self._verify) as client:
                resp = client.post(url, headers=headers, json=payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("NeMo guardrail checks request failed")
            return GuardrailsResult(
                blocked=True,
                status="error",
                detail=f"guardrails unreachable: {exc}",
                raw={},
            )

        if resp.status_code >= 400:
            logger.error(
                "NeMo guardrail checks HTTP %s body=%r",
                resp.status_code,
                (resp.text or "")[:500],
            )
            return GuardrailsResult(
                blocked=True,
                status="error",
                detail=f"guardrails HTTP {resp.status_code}",
                raw={},
            )

        data = resp.json() if resp.content else {}
        status = str(data.get("status") or "").lower()
        blocked = status == "blocked"
        activated = (
            ((data.get("guardrails_data") or {}).get("log") or {}).get(
                "activated_rails"
            )
            or []
        )
        detail = (
            f"blocked by: {', '.join(str(x) for x in activated)}"
            if blocked and activated
            else status or "ok"
        )
        logger.info(
            "NeMo guardrails RESULT status=%s blocked=%s activated=%s",
            status,
            blocked,
            activated,
        )
        return GuardrailsResult(
            blocked=blocked,
            status=status or "unknown",
            detail=detail,
            raw=data if isinstance(data, dict) else {},
        )
