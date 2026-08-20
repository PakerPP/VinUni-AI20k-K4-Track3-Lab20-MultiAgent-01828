"""Real tracing providers.

The lab requires a provider beyond ad-hoc JSON. Two are wired up:

- **OpenTelemetry** — the default, because it needs no account or API key. Spans go
  to an in-memory exporter plus an optional console/OTLP exporter, so a run is
  verifiable offline.
- **LangSmith** — enabled automatically when `LANGSMITH_API_KEY` is configured.

Both are optional at import time: a missing package degrades to a no-op rather than
breaking the workflow.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)


class TraceProvider(Protocol):
    """Minimal provider contract used by `trace_span`."""

    name: str

    def start(self, name: str, attributes: dict[str, Any]) -> Any: ...

    def end(self, handle: Any, span: dict[str, Any]) -> None: ...


class NoOpProvider:
    """Fallback when no provider is configured or available."""

    name = "noop"

    def start(self, name: str, attributes: dict[str, Any]) -> Any:
        return None

    def end(self, handle: Any, span: dict[str, Any]) -> None:
        return None


class OpenTelemetryProvider:
    """OTel spans; works without any credentials."""

    name = "opentelemetry"

    def __init__(self, service_name: str = "multi-agent-research-lab") -> None:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        # Only install a provider once per process.
        current = trace.get_tracer_provider()
        if not isinstance(current, TracerProvider):
            provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
            trace.set_tracer_provider(provider)
        else:
            provider = current

        self.exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = trace.get_tracer(service_name)

    def start(self, name: str, attributes: dict[str, Any]) -> Any:
        span = self.tracer.start_span(name)
        for key, value in (attributes or {}).items():
            if isinstance(value, (str, bool, int, float)):
                span.set_attribute(key, value)
        return span

    def end(self, handle: Any, span: dict[str, Any]) -> None:
        if handle is None:
            return
        for key, value in (span.get("attributes") or {}).items():
            if isinstance(value, (str, bool, int, float)):
                handle.set_attribute(key, value)
        if span.get("status") == "error":
            handle.set_attribute("error", True)
            handle.set_attribute("error.message", str(span.get("error", ""))[:500])
        handle.end()

    def finished_spans(self) -> list[Any]:
        """Exported spans — used by tests and the CLI to prove the provider ran."""

        return list(self.exporter.get_finished_spans())


class LangSmithProvider:
    """LangSmith run tree; requires LANGSMITH_API_KEY."""

    name = "langsmith"

    def __init__(self, project: str) -> None:
        from langsmith import Client

        self.client = Client()
        self.project = project

    def start(self, name: str, attributes: dict[str, Any]) -> Any:
        from langsmith.run_trees import RunTree

        run = RunTree(
            name=name,
            run_type="chain",
            inputs=dict(attributes or {}),
            project_name=self.project,
        )
        run.post()
        return run

    def end(self, handle: Any, span: dict[str, Any]) -> None:
        if handle is None:
            return
        handle.end(outputs=dict(span.get("attributes") or {}))
        handle.patch()


def build_provider(kind: str | None = None) -> TraceProvider:
    """Pick a provider: explicit `kind`, else LangSmith if keyed, else OTel, else no-op."""

    settings = get_settings()
    kind = (kind or "auto").lower()

    if kind in {"auto", "langsmith"} and settings.langsmith_api_key:
        try:
            return LangSmithProvider(settings.langsmith_project)
        except Exception as exc:
            logger.warning("langsmith provider unavailable (%s)", exc)
            if kind == "langsmith":
                return NoOpProvider()

    if kind in {"auto", "opentelemetry", "otel"}:
        try:
            return OpenTelemetryProvider()
        except Exception as exc:
            logger.warning("opentelemetry provider unavailable (%s)", exc)

    return NoOpProvider()
