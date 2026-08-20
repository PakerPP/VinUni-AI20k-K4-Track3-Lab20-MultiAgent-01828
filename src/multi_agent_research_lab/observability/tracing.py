"""Tracing hooks.

Provider-agnostic by design. Spans are collected in-process and can be exported as
JSON (always available) or forwarded to LangSmith when a key is configured.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.providers import (
    NoOpProvider,
    TraceProvider,
    build_provider,
)

logger = logging.getLogger(__name__)

# Process-local span buffer; exported per run.
_SPANS: list[dict[str, Any]] = []
_PROVIDER: TraceProvider = NoOpProvider()


def configure_tracing(kind: str | None = None) -> TraceProvider:
    """Install a real tracing provider (OpenTelemetry by default)."""

    global _PROVIDER
    _PROVIDER = build_provider(kind)
    logger.info("tracing.provider=%s", _PROVIDER.name)
    return _PROVIDER


def get_provider() -> TraceProvider:
    """Return the active provider."""

    return _PROVIDER


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Record a timed span and append it to the process trace buffer."""

    started = perf_counter()
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "ok",
    }
    handle = _PROVIDER.start(name, span["attributes"])
    try:
        yield span
    except Exception as exc:
        span["status"] = "error"
        span["error"] = str(exc)
        raise
    finally:
        span["duration_seconds"] = perf_counter() - started
        span["provider"] = _PROVIDER.name
        _SPANS.append(span)
        try:
            _PROVIDER.end(handle, span)
        except Exception as exc:  # tracing must never break the workflow
            logger.warning("tracing provider failed to close span: %s", exc)
        logger.debug("span %s took %.3fs", name, span["duration_seconds"])


def get_spans() -> list[dict[str, Any]]:
    """Return a copy of collected spans."""

    return list(_SPANS)


def reset_spans() -> None:
    """Clear the span buffer (call between benchmark runs)."""

    _SPANS.clear()


def export_trace_json(
    state: ResearchState,
    path: Path,
    run_name: str = "run",
) -> Path:
    """Write a self-contained JSON trace: spans + state events + routing."""

    spans = get_spans()
    payload: dict[str, Any] = {
        "run_name": run_name,
        "exported_at": datetime.now(UTC).isoformat(),
        "query": state.request.query,
        "route_history": state.route_history,
        "iterations": state.iteration,
        "errors": state.errors,
        "spans": spans,
        "state_events": state.trace,
        "agent_results": [
            {
                "agent": str(r.agent),
                "metadata": r.metadata,
                "content_preview": r.content[:500],
            }
            for r in state.agent_results
        ],
        "sources": [
            {
                "title": d.title,
                "source_id": d.metadata.get("source_id"),
                "is_synthetic": d.metadata.get("is_synthetic"),
            }
            for d in state.sources
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("trace.exported path=%s spans=%d", path, len(spans))
    return path


def render_trace_table(state: ResearchState) -> str:
    """Human-readable trace for the terminal / report screenshots."""

    lines = [f"{'STEP':<4} {'AGENT':<12} {'EVENT':<22} DETAIL"]
    for i, event in enumerate(state.trace, 1):
        name = event.get("name", "")
        agent = name.split(".")[0]
        payload = event.get("payload", {})
        detail = ", ".join(f"{k}={v}" for k, v in list(payload.items())[:3])
        lines.append(f"{i:<4} {agent:<12} {name:<22} {detail[:80]}")
    return "\n".join(lines)
