"""Tests for LLM-as-judge scoring."""

from multi_agent_research_lab.core.errors import AgentExecutionError
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.judge import AnswerJudge, parse_verdict


class _FakeLLM:
    def __init__(self, reply: str = "SCORE: 7.5\nSTRENGTHS: clear\nWEAKNESSES: thin") -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str):
        self.calls += 1

        class _R:
            content = self.reply

        return _R()


class _BrokenLLM:
    def complete(self, system_prompt: str, user_prompt: str):
        raise AgentExecutionError("provider down")


def _state(answer: str | None = "# Answer\n\nGrounded [a1].") -> ResearchState:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    state.sources = [
        SourceDocument(title="D", snippet="s", metadata={"source_id": "a1"}),
    ]
    state.final_answer = answer
    return state


def test_parses_well_formed_verdict() -> None:
    verdict = parse_verdict("SCORE: 8.5\nSTRENGTHS: solid\nWEAKNESSES: terse")
    assert verdict is not None
    assert verdict.score == 8.5
    assert verdict.strengths == "solid"
    assert verdict.weaknesses == "terse"


def test_parses_verdict_surrounded_by_prose() -> None:
    verdict = parse_verdict("Here is my grade.\nSCORE: 6\nSTRENGTHS: ok\nWEAKNESSES: vague\nDone.")
    assert verdict is not None
    assert verdict.score == 6.0


def test_unparseable_verdict_returns_none() -> None:
    assert parse_verdict("I think it is quite good, maybe an eight.") is None


def test_score_is_clamped_to_range() -> None:
    assert parse_verdict("SCORE: 99").score == 10.0
    assert parse_verdict("SCORE: -4").score == 0.0


def test_judge_scores_state() -> None:
    llm = _FakeLLM()
    verdict = AnswerJudge(llm).score(_state())
    assert verdict.score == 7.5
    assert llm.calls == 1


def test_empty_answer_scores_zero_without_calling_llm() -> None:
    llm = _FakeLLM()
    verdict = AnswerJudge(llm).score(_state(answer=None))
    assert verdict.score == 0.0
    assert llm.calls == 0


def test_judge_failure_returns_none_rather_than_raising() -> None:
    assert AnswerJudge(_BrokenLLM()).score(_state()) is None
