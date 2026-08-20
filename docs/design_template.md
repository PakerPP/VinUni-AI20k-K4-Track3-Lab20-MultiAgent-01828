# Design Template

> **Đã điền** cho implementation trong `src/multi_agent_research_lab/`.
> Bản phân tích dài hơn (kèm lý do lựa chọn và số liệu): [`docs/design.md`](design.md).

## Problem

Trả lời câu hỏi nghiên cứu mở bằng báo cáo có dẫn chứng, mỗi claim truy được về một
`source_id` cụ thể. Retrieval chạy offline trên corpus `ai_agent_offline_research_corpus_v2/`
(30 topic), web search bị tắt theo benchmark rule của corpus.

## Why multi-agent?

Single-agent baseline thất bại ở **grounding**, không phải ở khả năng viết: đo thật được
citation coverage 0% vs 87% của multi-agent. Nguyên nhân là baseline không có bước
retrieval nào, nên không có gì để trích dẫn. Tách vai trò giải quyết đúng chỗ đó, và
tách verification ra agent riêng vì một model không đáng tin khi tự chấm citation của chính nó.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Chọn route kế tiếp, enforce guardrail | Toàn bộ `ResearchState` | `route_history`, `iteration` | Loop vô hạn nếu thiếu iteration guard |
| Researcher | Retrieve + cô đọng evidence | `query`, `max_sources` | `sources`, `research_notes` | Search fail; corpus không có topic khớp |
| Analyst | Cấu trúc hóa thành 5 mục insight | `research_notes`, fact bank | `analysis_notes` | Bịa claim ngoài phạm vi notes |
| Writer | Tổng hợp báo cáo có citation | notes + analysis + allow-list id | `final_answer` | Bịa `source_id` không tồn tại |
| Critic | Verify citation (deterministic) | `final_answer`, `sources` | Coverage report | Chỉ bắt lỗi id, không bắt lỗi ngữ nghĩa |

## Shared state

`ResearchState` (`core/state.py`):

| Field | Lý do cần |
|---|---|
| `request` | Query + `max_sources` + audience, bất biến suốt run |
| `iteration`, `route_history` | Guardrail chống loop; đồng thời là trace giải thích "ai làm gì" |
| `sources` | Evidence gốc; metadata giữ `source_id`, `is_synthetic`, `full_text` |
| `research_notes` / `analysis_notes` / `final_answer` | Ba mốc handoff — supervisor route dựa trên field nào còn `None` |
| `agent_results` | Output + metadata từng agent, phục vụ benchmark |
| `trace` | Event log có cấu trúc, export JSON |
| `errors` | Prefix `"{agent}: ..."` để supervisor đếm failure theo từng agent |

## Routing policy

`SupervisorAgent.decide()` — tất định, đọc trực tiếp từ state, không tốn LLM call:

```text
iteration >= max_iterations           -> done       (guardrail thắng mọi điều kiện khác)
chưa có sources                       -> researcher
có sources, chưa có analysis_notes    -> analyst
có analysis, chưa có final_answer     -> writer
đã có answer, critic chưa chạy        -> critic
còn lại                               -> done
```

Mỗi nhánh kiểm tra thêm `_blocked()`: agent fail >= 2 lần thì bị skip thay vì retry mãi.

## Guardrails

- **Max iterations:** `MAX_ITERATIONS=6`, check đầu tiên trong `decide()`; LangGraph có thêm `recursion_limit`.
- **Timeout:** `TIMEOUT_SECONDS=60`, truyền vào OpenAI client.
- **Retry:** tenacity, 3 lần, exponential backoff 1-8s, chỉ ở tầng `LLMClient`.
- **Fallback:** mỗi worker có `_fallback_*()` tất định — LLM chết thì pipeline vẫn ra kết quả thay vì crash.
- **Validation:** Pydantic ở mọi I/O; Critic verify citation; hallucinated citation bị trừ điểm quality.

## Benchmark plan

3 query trong `configs/lab_default.yaml`, mỗi query chạy cả 2 chế độ.

| Query | Metric | Expected outcome |
|---|---|---|
| GraphRAG state-of-the-art | coverage, quality | Multi-agent cao hơn rõ vì baseline không retrieval |
| Single vs multi cho customer support | coverage, cost | Multi-agent thắng chất lượng, thua cost |
| Production guardrails cho LLM agents | latency, coverage | Multi-agent chậm hơn ~2×, coverage cao hơn |

Kết quả thật (gpt-4o-mini): single-agent 10.9s / $0.0005 / quality 4.0 / coverage 0%;
multi-agent 22.7s / $0.0015 / quality 6.5 / coverage 87%.
