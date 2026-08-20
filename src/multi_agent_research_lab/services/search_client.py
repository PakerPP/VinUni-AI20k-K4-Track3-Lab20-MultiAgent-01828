"""Search client abstraction for ResearcherAgent.

The lab ships a self-contained offline corpus (`ai_agent_offline_research_corpus_v2/`),
so the default client retrieves evidence from it instead of calling a web API. This
matches the corpus benchmark rule: disable web search and cite embedded ids only.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_ROOT = Path("ai_agent_offline_research_corpus_v2")

_STOPWORDS = frozenset(
    # fmt: off
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "on",
        "with",
        "without",
        "vs",
        "versus",
        "how",
        "what",
        "when",
        "why",
        "which",
        "is",
        "are",
        "be",
        "do",
        "does",
        "using",
        "use",
        "used",
        "into",
        "from",
        "at",
        "by",
        "as",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "compare",
        "comparison",
        "summarize",
        "summary",
        "write",
        "report",
        "research",
        "state",
        "art",
    ]
    # fmt: on
)


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def _score(query_terms: set[str], text: str) -> float:
    """Overlap score normalized by query size, so longer docs do not automatically win."""

    if not query_terms:
        return 0.0
    doc_terms = _tokenize(text)
    return len(query_terms & doc_terms) / len(query_terms)


class SearchClient:
    """Offline corpus-backed search client."""

    def __init__(self, corpus_root: Path | None = None) -> None:
        self.corpus_root = Path(corpus_root) if corpus_root else DEFAULT_CORPUS_ROOT

    @property
    def topics_dir(self) -> Path:
        return self.corpus_root / "topics"

    def _topic_files(self) -> list[Path]:
        if not self.topics_dir.is_dir():
            raise AgentExecutionError(
                f"Offline corpus not found at {self.topics_dir}. "
                "Run from the repo root or pass corpus_root explicitly."
            )
        return sorted(self.topics_dir.glob("*.json"))

    def select_topic(self, query: str) -> Path:
        """Pick the corpus topic whose title/tags/question best match the query."""

        query_terms = _tokenize(query)
        best: tuple[float, Path] | None = None
        for path in self._topic_files():
            meta = _load_topic_index(path)
            score = _score(query_terms, meta)
            if best is None or score > best[0]:
                best = (score, path)
        if best is None:
            raise AgentExecutionError("Offline corpus contains no topic files.")
        logger.info("search.topic_selected file=%s score=%.3f", best[1].name, best[0])
        return best[1]

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return the most relevant embedded documents/articles for a query."""

        topic_path = self.select_topic(query)
        topic = _load_topic(topic_path)
        kb = topic.get("knowledge_base", {})
        query_terms = _tokenize(query)

        candidates: list[tuple[float, SourceDocument]] = []

        for doc in kb.get("source_documents", []):
            body = doc.get("full_text", "")
            text = f"{doc.get('title', '')} {body}"
            takeaways = doc.get("key_takeaways") or []
            snippet = takeaways[0] if takeaways else body[:400]
            candidates.append(
                (
                    _score(query_terms, text),
                    SourceDocument(
                        title=doc.get("title", "Untitled"),
                        url=doc.get("provenance_url"),
                        snippet=str(snippet)[:600],
                        metadata={
                            "source_id": doc.get("document_id"),
                            "citation_label": doc.get("citation_label"),
                            "is_synthetic": bool(doc.get("is_synthetic", False)),
                            "document_class": doc.get("document_class"),
                            "year": doc.get("year"),
                            "topic_id": topic.get("benchmark_metadata", {}).get("topic_id"),
                            "full_text": body,
                        },
                    ),
                )
            )

        for art in kb.get("knowledge_articles", []):
            body = art.get("content", "")
            candidates.append(
                (
                    _score(query_terms, f"{art.get('title', '')} {body}"),
                    SourceDocument(
                        title=art.get("title", "Untitled article"),
                        url=None,
                        snippet=body[:600],
                        metadata={
                            "source_id": art.get("article_id"),
                            "citation_label": art.get("article_id"),
                            "is_synthetic": False,
                            "document_class": "knowledge_article",
                            "topic_id": topic.get("benchmark_metadata", {}).get("topic_id"),
                            "full_text": body,
                        },
                    ),
                )
            )

        candidates.sort(key=lambda pair: pair[0], reverse=True)
        results = [doc for _, doc in candidates[:max_results]]
        logger.info("search.done query=%r results=%d", query[:60], len(results))
        return results

    def load_facts(self, query: str, limit: int = 12) -> list[dict[str, object]]:
        """Return fact-bank entries for the matched topic (used by the Analyst)."""

        topic = _load_topic(self.select_topic(query))
        facts = topic.get("knowledge_base", {}).get("fact_bank", [])
        return list(facts[:limit])


class MockSearchClient(SearchClient):
    """Fixed in-memory sources for tests that must not touch the corpus."""

    _FAKE_DOCS = [
        SourceDocument(
            title="Multi-Agent Coordination Overview",
            url="https://example.com/coordination",
            snippet="Specialization helps when subtasks need different information.",
            metadata={"source_id": "mock-1", "is_synthetic": True},
        ),
        SourceDocument(
            title="Cost of Orchestration",
            url="https://example.com/cost",
            snippet="Coordination overhead can erase quality gains on simple tasks.",
            metadata={"source_id": "mock-2", "is_synthetic": True},
        ),
        SourceDocument(
            title="Verification Agents",
            url="https://example.com/verify",
            snippet="Independent verification catches errors a single pass misses.",
            metadata={"source_id": "mock-3", "is_synthetic": True},
        ),
    ]

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        return list(self._FAKE_DOCS[:max_results])

    def load_facts(self, query: str, limit: int = 12) -> list[dict[str, object]]:
        return [
            {"fact_id": "M001", "statement": "Mock fact for tests.", "confidence": "low"},
        ]


@lru_cache(maxsize=64)
def _load_topic(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
        return data


@lru_cache(maxsize=64)
def _load_topic_index(path: Path) -> str:
    """Cheap match text (title + tags + research question) for topic selection."""

    topic = _load_topic(path).get("topic", {})
    tags = " ".join(topic.get("tags", []))
    return f"{topic.get('name', '')} {tags} {topic.get('research_question', '')} {path.stem}"
