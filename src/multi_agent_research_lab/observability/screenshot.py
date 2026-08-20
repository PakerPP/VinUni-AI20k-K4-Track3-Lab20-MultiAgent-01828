"""Render a run's trace to a PNG.

The lab deliverable asks for a screenshot of the trace. Generating it from the trace
data rather than photographing a terminal keeps it reproducible: re-run the command
and the image regenerates from the same source of truth.
"""

from __future__ import annotations

import logging
from pathlib import Path

from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

_AGENT_COLORS = {
    "supervisor": "#4C6EF5",
    "researcher": "#2F9E44",
    "analyst": "#F08C00",
    "writer": "#9C36B5",
    "critic": "#E03131",
    "workflow": "#495057",
    "baseline": "#1098AD",
}


def render_trace_png(
    state: ResearchState,
    path: Path,
    title: str = "Multi-Agent Run Trace",
) -> Path:
    """Draw the trace as a labelled timeline; returns the written path."""

    import matplotlib

    matplotlib.use("Agg")  # headless: no display needed
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    events = state.trace or []
    if not events:
        raise ValueError("state has no trace events to render")

    fig_height = max(3.0, 0.42 * len(events) + 2.2)
    fig, ax = plt.subplots(figsize=(11, fig_height))

    seen: dict[str, str] = {}
    for row, event in enumerate(events):
        name = str(event.get("name", ""))
        agent = name.split(".")[0]
        color = _AGENT_COLORS.get(agent, "#868E96")
        seen[agent] = color
        y = len(events) - row

        ax.barh(y, 1.0, height=0.62, color=color, alpha=0.85)
        ax.text(1.08, y, name, va="center", ha="left", fontsize=9, fontweight="medium")

        payload = event.get("payload", {}) or {}
        detail = ", ".join(f"{k}={v}" for k, v in list(payload.items())[:3])
        if detail:
            ax.text(3.15, y, detail[:78], va="center", ha="left", fontsize=8, color="#495057")
        ax.text(0.5, y, str(row + 1), va="center", ha="center", fontsize=8, color="white")

    ax.set_xlim(0, 9)
    ax.set_ylim(0.3, len(events) + 0.9)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    subtitle = (
        f"query: {state.request.query[:70]}\n"
        f"routes: {' > '.join(state.route_history) or 'n/a'}  |  "
        f"iterations: {state.iteration}  |  sources: {len(state.sources)}  |  "
        f"errors: {len(state.errors)}"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, loc="left", pad=14)
    ax.legend(
        handles=[mpatches.Patch(color=c, label=a) for a, c in sorted(seen.items())],
        loc="lower right",
        frameon=False,
        fontsize=8,
        ncol=len(seen),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("trace.png_written path=%s events=%d", path, len(events))
    return path
