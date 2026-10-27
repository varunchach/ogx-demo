"""Reference-free RAG evaluation (no ground-truth document).

Uses the cluster LLM via OGX to score faithfulness and answer relevancy from
the user question, retrieved contexts, and the model answer. Results are meant
for the in-app Evaluations panel and MLflow — not a static golden_qa file.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.ogx_client import OgxClient

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a strict RAG quality judge. Score ONLY from the provided "
    "question, retrieved contexts, and answer. Do not invent facts. "
    "Reply with a single JSON object and no markdown."
)


def avg_retrieval_score(scores: list[float]) -> float | None:
    """Mean file_search / retrieve score in [0, 1] when scores look normalized."""
    nums = [float(s) for s in scores if s is not None]
    if not nums:
        return None
    mean = sum(nums) / len(nums)
    # OGX may return raw distances; clip only when already in [0, 1]-ish range.
    if 0.0 <= mean <= 1.0:
        return round(mean, 4)
    return round(mean, 4)


def evaluate_turn(
    ogx: OgxClient,
    *,
    question: str,
    answer: str,
    contexts: list[str],
    retrieval_scores: list[float] | None = None,
) -> dict[str, Any]:
    """Return RAGAS-style reference-free scores in [0, 1] via OGX LLM-as-judge."""
    ctx_blocks = []
    for i, c in enumerate(contexts[:6], start=1):
        text = (c or "").strip()
        if text:
            ctx_blocks.append(f"[{i}] {text[:1200]}")
    joined = "\n\n".join(ctx_blocks) if ctx_blocks else "(no retrieved contexts)"

    prompt = (
        "Score this RAG turn using RAGAS-style reference-free metrics "
        "(no ground-truth document). Metrics (0.0 to 1.0):\n"
        "- faithfulness: fraction of the answer supported by the contexts "
        "(1 = fully supported, 0 = unsupported / hallucinated).\n"
        "- answer_relevancy: how well the answer addresses the question "
        "(1 = on-topic and useful, 0 = irrelevant).\n"
        "- context_sufficiency: were the retrieved contexts enough to answer "
        "(1 = sufficient, 0 = missing key facts).\n"
        'Return JSON: {"faithfulness":0.0,"answer_relevancy":0.0,'
        '"context_sufficiency":0.0,"notes":"short reason"}\n\n'
        f"QUESTION:\n{question}\n\n"
        f"CONTEXTS:\n{joined}\n\n"
        f"ANSWER:\n{answer}\n"
    )

    raw = ogx.chat(prompt, history=None, system=_SYSTEM)
    parsed = _parse_scores(raw)
    parsed["method"] = "ogx_llm_judge_reference_free"
    parsed["context_count"] = len(ctx_blocks)
    parsed["avg_retrieval_score"] = avg_retrieval_score(retrieval_scores or [])
    parsed["providers_hint"] = ["ragas-style", "llm-as-judge", "ogx"]
    parsed["skipped"] = False
    logger.info(
        "rag_eval faithfulness=%.3f relevancy=%.3f sufficiency=%.3f "
        "avg_retrieval=%s contexts=%s",
        parsed["faithfulness"],
        parsed["answer_relevancy"],
        parsed["context_sufficiency"],
        parsed["avg_retrieval_score"],
        parsed["context_count"],
    )
    return parsed


def _parse_scores(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    # Strip optional ```json fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)

    data: dict[str, Any]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("rag_eval JSON parse failed; raw=%r", raw[:400])
        data = {}

    def _clip(key: str, default: float = 0.0) -> float:
        try:
            return max(0.0, min(1.0, float(data.get(key, default))))
        except (TypeError, ValueError):
            return default

    return {
        "faithfulness": _clip("faithfulness"),
        "answer_relevancy": _clip("answer_relevancy"),
        "context_sufficiency": _clip("context_sufficiency"),
        "notes": str(data.get("notes") or "")[:500],
        "raw_preview": (raw or "")[:400],
    }
