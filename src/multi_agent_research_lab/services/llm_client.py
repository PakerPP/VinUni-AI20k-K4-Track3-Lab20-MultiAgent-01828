"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
Retry, timeout, and token accounting live here so agents stay free of provider details.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import AgentExecutionError

logger = logging.getLogger(__name__)

# USD per 1M tokens. Keeps cost estimation honest without an external pricing call.
_PRICING_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
}
_DEFAULT_PRICING = (0.15, 0.60)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate call cost in USD from token counts."""

    price_in, price_out = _PRICING_USD_PER_MTOK.get(model, _DEFAULT_PRICING)
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


class LLMClient:
    """Provider-agnostic LLM client backed by the OpenAI chat completions API."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        timeout_seconds: int | None = None,
        max_attempts: int = 3,
    ) -> None:
        settings = get_settings()
        self.model = model or settings.openai_model
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds or settings.timeout_seconds
        self.max_attempts = max_attempts
        self._api_key = settings.openai_api_key
        self._client: object | None = None

        # Accumulated across the process so benchmarks can report real cost.
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.call_count = 0

    def _ensure_client(self) -> object:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise AgentExecutionError(
                "OPENAI_API_KEY is not set. Add it to .env or use MockLLMClient for offline runs."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise AgentExecutionError(
                "openai package missing. Install with: pip install -e '.[llm]'"
            ) from exc

        self._client = OpenAI(api_key=self._api_key, timeout=self.timeout_seconds)
        return self._client

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion, retrying transient provider failures."""

        client = self._ensure_client()

        @retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        def _call() -> object:
            return client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                temperature=self.temperature,
                timeout=self.timeout_seconds,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

        try:
            response = _call()
        except Exception as exc:
            raise AgentExecutionError(
                f"LLM call failed after {self.max_attempts} attempts: {exc}"
            ) from exc

        content = (response.choices[0].message.content or "").strip()  # type: ignore[attr-defined]
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)

        cost = None
        if input_tokens is not None and output_tokens is not None:
            cost = estimate_cost(self.model, input_tokens, output_tokens)
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += cost
        self.call_count += 1

        logger.info(
            "llm.complete model=%s in_tokens=%s out_tokens=%s cost_usd=%s",
            self.model,
            input_tokens,
            output_tokens,
            f"{cost:.6f}" if cost is not None else None,
        )
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )


class MockLLMClient(LLMClient):
    """Deterministic offline client so the workflow runs without an API key.

    This is not a language model: it extracts the citation ids and question present in
    the prompt and echoes them back in the expected shape. That is enough for the
    pipeline, the citation checker, and the benchmark to exercise real code paths
    offline — but quality numbers from a mock run compare plumbing, not writing.
    """

    _ROLE_RE = re.compile(r"You are (?:a|an) (\w+)", re.IGNORECASE)
    _CITE_RE = re.compile(r"\[([A-Za-z0-9_\-]{1,40})\]")
    # Bracketed words that are labels, not citation ids.
    _NON_CITATION_TOKENS = frozenset({"synthetic", "sic", "source_id", "fact_id"})

    def __init__(self, model: str = "mock-model", temperature: float = 0.0) -> None:
        super().__init__(model=model, temperature=temperature)

    def _extract_question(self, user_prompt: str) -> str:
        for line in user_prompt.splitlines():
            if line.lower().startswith(("research question:", "question:")):
                return line.split(":", 1)[1].strip()
        return user_prompt.strip().splitlines()[0][:120] if user_prompt.strip() else ""

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        role_match = self._ROLE_RE.search(system_prompt)
        role = role_match.group(1).lower() if role_match else "assistant"
        question = self._extract_question(user_prompt)

        # Reuse the evidence ids supplied by the caller so citations stay grounded.
        # Prefer the explicit allow-list the writer prompt carries; otherwise scan,
        # skipping bracketed words that are metadata labels rather than citations.
        allowed_line = next(
            (
                line
                for line in user_prompt.splitlines()
                if line.lower().startswith("allowed citation ids:")
            ),
            None,
        )
        scan_target = allowed_line if allowed_line else user_prompt
        cites = [
            cid
            for cid in dict.fromkeys(self._CITE_RE.findall(scan_target))
            if cid.lower() not in self._NON_CITATION_TOKENS
        ]

        if role == "research":
            bullets = [
                f"- Evidence point {n} drawn from the supplied corpus [{cid}]"
                for n, cid in enumerate(cites, 1)
            ]
            body = "\n".join(bullets) if bullets else "- No evidence ids supplied."
            content = f"Research notes (mock) for: {question}\n{body}"
        elif role == "analyst":
            joined = ", ".join(f"[{c}]" for c in cites) or "(none)"
            content = (
                "1. Key claims\n"
                f"   - Claims consolidated from {joined}\n"
                "2. Points of agreement\n"
                "   - Sources broadly align on the framing of the question.\n"
                "3. Tensions or contradictions\n"
                "   - Not derivable from a mock model; needs a real LLM.\n"
                "4. Weak evidence\n"
                "   - Synthetic items are flagged upstream by the researcher.\n"
                "5. What is still missing\n"
                "   - Quantitative comparison across sources."
            )
        else:  # writer or single-agent baseline
            sections = "\n\n".join(
                f"## Section {n}\n\n- Grounded statement supported by [{cid}]"
                for n, cid in enumerate(cites, 1)
            )
            content = (
                f"# {question or 'Research answer'} (mock)\n\n"
                "Mock synthesis produced without a live model.\n\n"
                f"{sections or '- No evidence available to cite.'}\n\n"
                "## Conclusion\n\n- Replace the mock with a real LLM for a meaningful answer."
            )

        input_tokens = max(1, len(system_prompt) + len(user_prompt)) // 4
        output_tokens = max(1, len(content)) // 4
        cost = estimate_cost("gpt-4o-mini", input_tokens, output_tokens)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.call_count += 1
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
