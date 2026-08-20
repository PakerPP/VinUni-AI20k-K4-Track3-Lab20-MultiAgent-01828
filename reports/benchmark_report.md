# Benchmark Report: Single-Agent vs Multi-Agent

_Generated: 2026-08-20 11:43 UTC_

## 1. Setup

- **Evidence source:** offline corpus `ai_agent_offline_research_corpus_v2/` (30 topics, web search disabled per the corpus benchmark rule).
- **Baseline (control):** one LLM call over the **same evidence** as the crew - so the comparison measures orchestration, not retrieval.
- **Baseline (no retrieval):** contrast arm with no evidence at all; its 0% citation coverage is an artefact of having nothing to cite.
- **Multi-agent:** Supervisor routes Researcher → Analyst → Writer → Critic.
- **Citation rule:** answers may cite only `source_id` / `article_id` values present in the retrieved evidence.

**Queries benchmarked:**

1. Research GraphRAG state-of-the-art and write a 500-word summary
2. Compare single-agent and multi-agent workflows for customer support
3. Summarize production guardrails for LLM agents

## 2. Results

| Run | Latency (s) | Cost (USD) | Quality | Citation cov. | Failure rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| baseline-1 | 11.61 | 0.0007 | 7.5 | 100% | 0% | routes=single_agent; sources=5; llm_calls=1; grader=llm-judge; hallucinated_citations=0; errors=0; judge_note=Some claims lack direct citations, and the discussion could benefit from a more  |
| multi-agent-1 | 19.85 | 0.0015 | 7.5 | 80% | 0% | routes=researcher>analyst>writer>critic>done; sources=5; llm_calls=3; grader=llm-judge; hallucinated_citations=0; errors=0; judge_note=It lacks a critical assessment of the cited evidence and does not sufficiently a |
| baseline-noretrieval-1 | 9.79 | 0.0005 | 7.5 | 0% | 0% | routes=single_agent; sources=0; llm_calls=1; grader=llm-judge; hallucinated_citations=0; errors=0; judge_note=Lacks specific evidence or citations to support claims made about the framework' |
| baseline-2 | 6.72 | 0.0006 | 8.5 | 80% | 0% | routes=single_agent; sources=5; llm_calls=1; grader=llm-judge; hallucinated_citations=0; errors=0; judge_note=It could better acknowledge the limitations of the evidence and the uncertainty  |
| multi-agent-2 | 21.06 | 0.0016 | 8.5 | 60% | 0% | routes=researcher>analyst>writer>critic>done; sources=5; llm_calls=3; grader=llm-judge; hallucinated_citations=0; errors=0; judge_note=It could better acknowledge the limitations of the evidence and the contexts in  |
| baseline-noretrieval-2 | 10.27 | 0.0004 | 8.5 | 0% | 0% | routes=single_agent; sources=0; llm_calls=1; grader=llm-judge; hallucinated_citations=0; errors=0; judge_note=Lacks cited evidence to support claims and does not acknowledge any uncertaintie |
| baseline-3 | 11.23 | 0.0007 | 8.5 | 40% | 0% | routes=single_agent; sources=5; llm_calls=1; grader=llm-judge; hallucinated_citations=0; errors=0; judge_note=Some sections could benefit from clearer connections between claims and their su |
| multi-agent-3 | 19.36 | 0.0014 | 8.5 | 80% | 0% | routes=researcher>analyst>writer>critic>done; sources=5; llm_calls=3; grader=llm-judge; hallucinated_citations=0; errors=0; judge_note=Some claims could benefit from more empirical evidence rather than relying on sy |
| baseline-noretrieval-3 | 9.74 | 0.0004 | 8.5 | 0% | 0% | routes=single_agent; sources=0; llm_calls=1; grader=llm-judge; hallucinated_citations=0; errors=0; judge_note=Lacks specific examples or citations to support claims made. |

## 3. Aggregate comparison

| Metric | Single-agent | Multi-agent | Delta | No-retrieval |
|---|---:|---:|---:|---:|
| Avg latency (s) | 9.85 | 20.09 | +10.23 | 9.93 |
| Avg cost (USD) | 0.0007 | 0.0015 | +0.0008 | 0.0004 |
| Avg quality (0-10) | 8.2 | 8.2 | +0.0 | 8.2 |
| Avg citation coverage | 73% | 73% | +0% | 0% |

> The **no-retrieval** column is the arm an earlier version of this benchmark
> used as its control. Comparing the crew against it produces a large citation
> gap that measures retrieval rather than orchestration. It is reported here
> only to show why that comparison is misleading.

## 4. Trace summary

### baseline-1

- Routes: `single_agent`
- Iterations: 1
- Sources retrieved: 5
- Errors: none
- Evidence ids: autogen, metagpt, anthropic_agents, agentbench, llm_agents_blog

### multi-agent-1

- Routes: `researcher → analyst → writer → critic → done`
- Iterations: 5
- Sources retrieved: 5
- Errors: none
- Evidence ids: autogen, metagpt, anthropic_agents, agentbench, llm_agents_blog

### baseline-noretrieval-1

- Routes: `single_agent`
- Iterations: 1
- Sources retrieved: 0
- Errors: none

### baseline-2

- Routes: `single_agent`
- Iterations: 1
- Sources retrieved: 5
- Errors: none
- Evidence ids: anthropic_agents, gaia, T01-SYN-A, autogen, metagpt
- Synthetic (fictional) evidence present: T01-SYN-A

### multi-agent-2

- Routes: `researcher → analyst → writer → critic → done`
- Iterations: 5
- Sources retrieved: 5
- Errors: none
- Evidence ids: anthropic_agents, gaia, T01-SYN-A, autogen, metagpt
- Synthetic (fictional) evidence present: T01-SYN-A

### baseline-noretrieval-2

- Routes: `single_agent`
- Iterations: 1
- Sources retrieved: 0
- Errors: none

### baseline-3

- Routes: `single_agent`
- Iterations: 1
- Sources retrieved: 5
- Errors: none
- Evidence ids: autogen, camel, A02, A07, chatdev

### multi-agent-3

- Routes: `researcher → analyst → writer → critic → done`
- Iterations: 5
- Sources retrieved: 5
- Errors: none
- Evidence ids: autogen, camel, A02, A07, chatdev

### baseline-noretrieval-3

- Routes: `single_agent`
- Iterations: 1
- Sources retrieved: 0
- Errors: none

