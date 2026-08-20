"""Tests for offline corpus retrieval."""

from pathlib import Path

import pytest

from multi_agent_research_lab.services.search_client import SearchClient

CORPUS = Path("ai_agent_offline_research_corpus_v2")
requires_corpus = pytest.mark.skipif(
    not (CORPUS / "topics").is_dir(), reason="offline corpus not present"
)


@requires_corpus
def test_selects_topic_matching_the_query() -> None:
    client = SearchClient()
    picked = client.select_topic("prompt injection and tool poisoning against agents")
    assert "prompt_injection" in picked.name


@requires_corpus
def test_search_returns_cited_sources() -> None:
    client = SearchClient()
    docs = client.search("long term memory architectures for agents", max_results=4)
    assert len(docs) == 4
    for doc in docs:
        assert doc.metadata.get("source_id")
        assert doc.snippet


@requires_corpus
def test_search_respects_max_results() -> None:
    assert len(SearchClient().search("multi-agent coordination", max_results=2)) == 2


@requires_corpus
def test_load_facts_returns_fact_bank_entries() -> None:
    facts = SearchClient().load_facts("multi-agent architectures", limit=5)
    assert len(facts) == 5
    assert all("statement" in f for f in facts)


def test_missing_corpus_raises_actionable_error() -> None:
    from multi_agent_research_lab.core.errors import AgentExecutionError

    client = SearchClient(corpus_root=Path("does-not-exist"))
    with pytest.raises(AgentExecutionError, match="corpus not found"):
        client.search("anything")
