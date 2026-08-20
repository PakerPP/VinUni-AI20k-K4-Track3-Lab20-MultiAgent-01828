"""Tests for tracing providers and trace export."""

import json

import pytest

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.providers import (
    NoOpProvider,
    OpenTelemetryProvider,
    build_provider,
)
from multi_agent_research_lab.observability.tracing import (
    configure_tracing,
    export_trace_json,
    get_spans,
    render_trace_table,
    reset_spans,
    trace_span,
)


@pytest.fixture(autouse=True)
def _clean_spans():
    reset_spans()
    yield
    reset_spans()


def test_otel_provider_exports_real_spans() -> None:
    provider = configure_tracing("otel")
    assert provider.name == "opentelemetry"
    with trace_span("unit.span", {"kind": "test"}) as span:
        span["attributes"]["count"] = 2

    exported = provider.finished_spans()
    assert any(s.name == "unit.span" for s in exported)
    attrs = dict(next(s for s in exported if s.name == "unit.span").attributes)
    assert attrs["kind"] == "test"
    assert attrs["count"] == 2


def test_span_records_duration_and_provider() -> None:
    configure_tracing("otel")
    with trace_span("timed"):
        pass
    span = get_spans()[-1]
    assert span["duration_seconds"] >= 0
    assert span["provider"] == "opentelemetry"
    assert span["status"] == "ok"


def test_span_marks_errors_and_reraises() -> None:
    configure_tracing("none")
    with pytest.raises(RuntimeError), trace_span("boom"):
        raise RuntimeError("kaboom")
    span = get_spans()[-1]
    assert span["status"] == "error"
    assert "kaboom" in span["error"]


def test_provider_failure_does_not_break_workflow() -> None:
    """Tracing is best-effort: a broken provider must not propagate."""

    class BrokenProvider(NoOpProvider):
        name = "broken"

        def end(self, handle, span):
            raise RuntimeError("exporter down")

    import multi_agent_research_lab.observability.tracing as tracing

    tracing._PROVIDER = BrokenProvider()
    try:
        with trace_span("safe"):
            pass  # must not raise
    finally:
        tracing._PROVIDER = NoOpProvider()


def test_build_provider_falls_back_to_noop_when_disabled() -> None:
    assert build_provider("none").name == "noop"


def test_build_provider_auto_selects_otel_without_keys() -> None:
    assert isinstance(build_provider("auto"), OpenTelemetryProvider)


def test_export_trace_json_writes_full_payload(tmp_path) -> None:
    configure_tracing("otel")
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.record_route("researcher")
    state.add_trace_event("researcher.done", {"num_sources": 2})
    with trace_span("workflow.run"):
        pass

    path = export_trace_json(state, tmp_path / "trace.json", run_name="unit")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_name"] == "unit"
    assert payload["route_history"] == ["researcher"]
    assert any(s["name"] == "workflow.run" for s in payload["spans"])
    assert payload["state_events"][0]["name"] == "researcher.done"


def test_render_trace_table_lists_events() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.add_trace_event("writer.done", {"answer_chars": 10})
    table = render_trace_table(state)
    assert "writer.done" in table
    assert "answer_chars=10" in table


def test_cost_summary_reports_per_agent_spend() -> None:
    """The rubric asks "who spent what" - the trace must answer it."""

    from multi_agent_research_lab.core.schemas import AgentName, AgentResult

    configure_tracing("otel")
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.agent_results.append(
        AgentResult(
            agent=AgentName.RESEARCHER,
            content="notes",
            metadata={"input_tokens": 100, "output_tokens": 40, "cost_usd": 0.001},
        )
    )
    state.agent_results.append(
        AgentResult(agent=AgentName.SUPERVISOR, content="route=done", metadata={})
    )

    import json

    from multi_agent_research_lab.observability.tracing import export_trace_json

    with trace_span("researcher.run"):
        pass
    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as tmp:
        path = export_trace_json(state, _Path(tmp) / "t.json")
        summary = json.loads(path.read_text(encoding="utf-8"))["cost_summary"]

    assert summary["per_agent"]["researcher"]["cost_usd"] == 0.001
    assert summary["per_agent"]["researcher"]["input_tokens"] == 100
    # Deterministic agents cost nothing - that is the design point.
    assert summary["per_agent"]["supervisor"]["cost_usd"] == 0.0
    assert summary["total_cost_usd"] == 0.001
