# Lab 20: Multi-Agent Research System Starter

Bài lab **Multi-Agent Systems**: hệ thống nghiên cứu gồm **Supervisor + Researcher + Analyst + Writer + Critic**, benchmark với single-agent baseline.

> **Trạng thái: đã hoàn thành.** Toàn bộ `TODO(student)` trong `src/` đã được implement.
> Thiết kế và lý do lựa chọn: [`docs/design.md`](docs/design.md). Kết quả: [`reports/benchmark_report.md`](reports/benchmark_report.md).

**Điểm đặc thù:** retrieval chạy **offline** trên corpus `ai_agent_offline_research_corpus_v2/` (30 topic) thay vì web search — đúng benchmark rule của corpus, và không cần Tavily key.

## Learning outcomes

Sau 2 giờ lab, học viên cần có thể:

1. Thiết kế role rõ ràng cho nhiều agent.
2. Xây dựng shared state đủ thông tin cho handoff.
3. Thêm guardrail tối thiểu: max iterations, timeout, retry/fallback, validation.
4. Trace được luồng chạy và giải thích agent nào làm gì.
5. Benchmark single-agent vs multi-agent theo quality, latency, cost.

## Architecture mục tiêu

```text
User Query
   |
   v
Supervisor / Router
   |------> Researcher Agent  -> research_notes
   |------> Analyst Agent     -> analysis_notes
   |------> Writer Agent      -> final_answer
   |
   v
Trace + Benchmark Report
```

## Cấu trúc repo

```text
.
├── src/multi_agent_research_lab/
│   ├── agents/              # Agent interfaces + skeletons
│   ├── core/                # Config, state, schemas, errors
│   ├── graph/               # LangGraph workflow skeleton
│   ├── services/            # LLM, search, storage clients
│   ├── evaluation/          # Benchmark/evaluation skeleton
│   ├── observability/       # Logging/tracing hooks
│   └── cli.py               # CLI entrypoint
├── configs/                 # YAML configs for lab variants
├── docs/                    # Lab guide, rubric, design notes
├── tests/                   # Unit tests for skeleton behavior
├── notebooks/               # Optional notebook entrypoint
├── scripts/                 # Helper scripts
├── .env.example             # Environment variables template
├── pyproject.toml           # Python project config
├── Dockerfile               # Containerized dev/runtime
└── Makefile                 # Common commands
```

## Quickstart

### 1. Tạo môi trường

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev,llm]"
cp .env.example .env
```

### 2. Cấu hình API keys

Mở `.env` và điền key cần thiết.

```bash
OPENAI_API_KEY=...
# optional
LANGSMITH_API_KEY=...
TAVILY_API_KEY=...
```

### 3. Chạy smoke test

```bash
make test
python -m multi_agent_research_lab.cli --help
```

### 4. Chạy baseline skeleton

```bash
python -m multi_agent_research_lab.cli baseline \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Lệnh này chỉ chạy khung baseline tối giản. Học viên cần tự triển khai logic LLM thực tế trong `src/multi_agent_research_lab/services/llm_client.py`.

### 5. Chạy multi-agent skeleton

```bash
python -m multi_agent_research_lab.cli multi-agent \
  --query "Research GraphRAG state-of-the-art and write a 500-word summary"
```

Mặc định lệnh sẽ báo các `TODO` cần làm. Đây là chủ đích của starter repo.

## Milestones trong 2 giờ lab

| Thời lượng | Milestone | File gợi ý |
|---:|---|---|
| 0-15' | Setup, chạy baseline skeleton | `cli.py`, `services/llm_client.py` |
| 15-45' | Build Supervisor / router | `agents/supervisor.py`, `graph/workflow.py` |
| 45-75' | Thêm Researcher, Analyst, Writer | `agents/*.py`, `core/state.py` |
| 75-95' | Trace + benchmark single vs multi | `observability/tracing.py`, `evaluation/benchmark.py` |
| 95-115' | Peer review theo rubric | `docs/peer_review_rubric.md` |
| 115-120' | Exit ticket | `docs/lab_guide.md` |

## Quy ước production trong repo

- Tách rõ `agents`, `services`, `core`, `graph`, `evaluation`, `observability`.
- Không hard-code API key trong code.
- Tất cả input/output chính dùng Pydantic schema.
- Có type hints, linting, formatting, unit test tối thiểu.
- Có logging/tracing hook ngay từ đầu.
- Không để agent chạy vô hạn: dùng `max_iterations`, `timeout_seconds`.
- Có benchmark report thay vì chỉ demo output đẹp.

## Cách chạy

```bash
# Multi-agent, retrieval offline, xuất JSON trace
python -m multi_agent_research_lab.cli multi-agent   -q "When is a multi-agent architecture better than a single agent?"   --max-sources 4 --trace-out reports/trace_demo.json

# Single-agent baseline
python -m multi_agent_research_lab.cli baseline -q "..."

# Benchmark 3 arm (baseline / multi-agent / baseline không retrieval) + LLM judge
python -m multi_agent_research_lab.cli benchmark --out reports/benchmark_report.md

# Trace kèm screenshot PNG
python -m multi_agent_research_lab.cli multi-agent -q "..."   --tracer otel --trace-out reports/trace_demo.json   --screenshot reports/screenshots/trace.png
```

Baseline mặc định nhận **cùng evidence** như crew (`--no-retrieval` để chạy arm cũ).
Tắt judge bằng `--no-judge`, tắt arm thứ ba bằng `--no-naive`.

Chọn tracing provider bằng `--tracer otel|langsmith|auto|none` (mặc định `auto`:
LangSmith nếu có key, còn lại OpenTelemetry — không cần key).

Thêm `--mock` vào bất kỳ lệnh nào để chạy **không cần API key** (mock LLM tất định).
Nếu `OPENAI_API_KEY` trống, CLI tự động dùng mock và báo bằng dòng cảnh báo vàng.

## Kết quả benchmark

Chạy thật với `gpt-4o-mini`, 3 query, chấm bằng **LLM-as-judge**
(chi tiết: [`reports/benchmark_report.md`](reports/benchmark_report.md)):

| Metric | Single-agent | Multi-agent | Delta | No-retrieval |
|---|---:|---:|---:|---:|
| Latency (s) | 9.85 | 20.09 | **+104%** | 9.93 |
| Cost (USD) | 0.0007 | 0.0015 | **+114%** | 0.0004 |
| Quality (0-10) | 8.2 | 8.2 | **0.0** | 8.2 |
| Citation coverage | 73% | 73% | **0%** | 0% |

**Kết quả là âm, và đó chính là phát hiện.** Trên workload này multi-agent tốn gấp đôi
thời gian và gấp đôi tiền mà **không** tốt hơn: quality ngang nhau, citation coverage
ngang nhau. Lý do: retrieval ở đây quá dễ — corpus trả về đúng 5 nguồn ngay lần đầu, và
task chỉ là "đọc 5 tài liệu rồi tổng hợp". Một model đủ tốt làm xong trong một lượt;
mọi thứ crew thêm vào (handoff sang analyst, vòng critic, viết lại) là coordination
overhead không có việc để biện minh.

> **Vì sao con số này khác bản trước.** Phiên bản đầu cho baseline **không** retrieval,
> ra kết quả 0% vs 87% — nhưng đó là đo *retrieval*, không phải đo *orchestration*:
> baseline không có gì để trích dẫn thì coverage 0% là tất yếu. Cột "No-retrieval" giữ
> lại arm cũ để thấy rõ phép so sánh đó gây hiểu lầm thế nào.

## Implementation

| Thành phần | File | Ghi chú |
|---|---|---|
| Routing policy | `agents/supervisor.py` | State-driven, tất định; max-iteration + skip agent lỗi |
| Retrieval offline | `services/search_client.py` | Chọn topic theo term overlap, trả `source_id` để cite |
| LLM client | `services/llm_client.py` | Retry (tenacity), timeout, token + cost tracking |
| Citation verify | `agents/critic.py` | String matching, không dùng LLM tự chấm chính nó |
| Workflow | `graph/workflow.py` | LangGraph, fallback sang plain loop nếu langgraph lỗi |
| Baseline (control) | `evaluation/baseline.py` | Nhận **cùng evidence** như crew, 1 LLM call — cô lập biến orchestration |
| Quality grading | `evaluation/judge.py` | **LLM-as-judge** 0-10, heuristic làm fallback |
| Metrics | `evaluation/benchmark.py` | Coverage, hallucinated citations, LLM calls, cost |
| Trace | `observability/tracing.py`, `providers.py` | **OpenTelemetry** (mặc định, không cần key) + LangSmith nếu có key; export JSON + bảng terminal |

## Deliverables

1. Code: `src/multi_agent_research_lab/` — **53 test pass**, ruff clean.
2. Design doc: [`docs/design.md`](docs/design.md) + template đã điền [`docs/design_template.md`](docs/design_template.md).
3. Benchmark: [`reports/benchmark_report.md`](reports/benchmark_report.md).
4. Trace: `reports/trace_demo.json` + **screenshot** `reports/screenshots/*.png`
   (sinh bằng `--trace-out` và `--screenshot`, tái tạo được từ dữ liệu trace).
5. Failure mode + cách fix: [`docs/design.md`](docs/design.md) — **5 lỗi gặp thật** (gồm cả lỗi thiết kế phép đo) + 2 hạn chế còn tồn tại.

## References

- Anthropic: Building effective agents — https://www.anthropic.com/engineering/building-effective-agents
- OpenAI Agents SDK orchestration/handoffs — https://developers.openai.com/api/docs/guides/agents/orchestration
- LangGraph concepts — https://langchain-ai.github.io/langgraph/concepts/
- LangSmith tracing — https://docs.smith.langchain.com/
- Langfuse tracing — https://langfuse.com/docs
