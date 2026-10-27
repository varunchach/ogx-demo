"""OpenTelemetry → RHOAI data-science collector → Tempo."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

logger = logging.getLogger(__name__)

_provider: TracerProvider | None = None
_tracer: trace.Tracer | None = None

# Probe / static traffic must not flood Tempo (Observe → Traces).
# Explicit path regexes — substring "readyz" alone was not always honored.
_EXCLUDED_URL_REGEXES = (
    r"/readyz/?",
    r"/health/?",
    r"/livez/?",
    r"/config/?",
    r"/static",
    r"/favicon\.ico",
    r"/docs/?",
    r"/openapi\.json",
    r"/redoc/?",
)
_EXCLUDED_URLS = ",".join(_EXCLUDED_URL_REGEXES)
_PROBE_PATH_RE = re.compile(
    r"/(?:readyz|health|livez|config|favicon\.ico)(?:/|$|\?)",
    re.IGNORECASE,
)
_PROBE_NAME_RE = re.compile(
    r"(?:readyz|health|livez)\b|^GET$|^GET\s+/?(?:readyz|health|livez|config)?$",
    re.IGNORECASE,
)


class _DropProbeSpansExporter(SpanExporter):
    """Drop probe/static GET spans before they reach OTLP/Tempo."""

    def __init__(self, delegate: SpanExporter) -> None:
        self._delegate = delegate

    def export(self, spans: Any) -> SpanExportResult:
        kept: list[ReadableSpan] = []
        for span in spans:
            if _is_probe_span(span):
                continue
            kept.append(span)
        if not kept:
            return SpanExportResult.SUCCESS
        return self._delegate.export(kept)

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)


def _is_probe_span(span: ReadableSpan) -> bool:
    """True for kubelet probes and other noise (not POST /chat|/ingest)."""
    name = (span.name or "").strip()
    attrs = span.attributes or {}
    method = str(attrs.get("http.method") or attrs.get("http.request.method") or "")
    paths = [
        str(attrs.get(k) or "")
        for k in ("http.route", "http.target", "url.path", "http.url")
    ]
    joined = " ".join(paths)

    if _PROBE_NAME_RE.search(name) or _PROBE_PATH_RE.search(joined):
        return True
    # FastAPI sometimes names the span just "GET" with route in attributes.
    if method.upper() == "GET" and _PROBE_PATH_RE.search(joined or name):
        return True
    # Bare service:GET with no route and tiny duration → almost always a probe.
    if (
        method.upper() == "GET"
        and not any(paths)
        and re.search(r"(^|\s)GET$", name, re.I)
    ):
        return True
    return False


def setup_tracing(
    service_name: str,
    otlp_endpoint: str,
    protocol: str = "http/protobuf",
    sample_ratio: float = 1.0,
) -> trace.Tracer:
    """Configure OTLP export + FastAPI/httpx instrumentation."""
    global _provider, _tracer
    if _tracer is not None:
        return _tracer

    # Ensure instrumentors honor exclusions even if kwargs are ignored.
    os.environ["OTEL_PYTHON_EXCLUDED_URLS"] = _EXCLUDED_URLS
    os.environ["OTEL_PYTHON_FASTAPI_EXCLUDED_URLS"] = _EXCLUDED_URLS

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": "ogx-chatbot",
            "service.version": "1.4.20",
            "deployment.environment": "openshift-ai",
        }
    )
    ratio = max(0.0, min(1.0, sample_ratio))
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(ratio)),
    )

    if otlp_endpoint:
        endpoint = otlp_endpoint.rstrip("/")
        if protocol.startswith("grpc"):
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter as GrpcExporter,
            )

            exporter: SpanExporter = GrpcExporter(endpoint=endpoint, insecure=True)
            logger.info("OTLP gRPC traces -> %s", endpoint)
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter as HttpExporter,
            )

            traces_url = (
                endpoint
                if endpoint.endswith("/v1/traces")
                else f"{endpoint}/v1/traces"
            )
            exporter = HttpExporter(endpoint=traces_url)
            logger.info("OTLP HTTP traces -> %s", traces_url)
        provider.add_span_processor(
            BatchSpanProcessor(_DropProbeSpansExporter(exporter))
        )
    else:
        provider.add_span_processor(
            BatchSpanProcessor(_DropProbeSpansExporter(ConsoleSpanExporter()))
        )
        logger.warning("OTEL_EXPORTER_OTLP_ENDPOINT unset; spans go to console only")

    trace.set_tracer_provider(provider)
    _provider = provider
    _tracer = trace.get_tracer(service_name)
    return _tracer


def instrument_app(app: Any) -> None:
    """Instrument FastAPI + httpx; exclude /readyz and other probes from Tempo."""
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls=_EXCLUDED_URLS,
        )
        HTTPXClientInstrumentor().instrument()
        logger.info(
            "OTel instrumentation on (excluded_urls=%s; /readyz spans dropped)",
            _EXCLUDED_URLS,
        )
    except Exception:  # noqa: BLE001
        logger.exception("OTel FastAPI/httpx instrumentation failed")


def get_tracer() -> trace.Tracer:
    return _tracer or trace.get_tracer("ogx-chatbot")


def current_trace_id() -> str:
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx or not ctx.is_valid:
        return ""
    return format(ctx.trace_id, "032x")


def flush_traces(timeout_millis: int = 5000) -> None:
    if _provider is not None:
        _provider.force_flush(timeout_millis=timeout_millis)
