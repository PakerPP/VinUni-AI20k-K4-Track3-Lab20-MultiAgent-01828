# Design: Multi-Agent Research System

Bản điền của `docs/design_template.md` cho implementation trong `src/multi_agent_research_lab/`.

## Problem

Trả lời câu hỏi nghiên cứu mở (ví dụ "Khi nào multi-agent tốt hơn single-agent?") bằng
một báo cáo có dẫn chứng, mỗi claim truy được về nguồn cụ thể. Hệ thống chạy **offline**
trên corpus `ai_agent_offline_research_corpus_v2/` (30 topic, mỗi topic ~4,900 từ prose,
9 source documents, 30 atomic facts) — web search bị tắt theo đúng benchmark rule của corpus.

## Why multi-agent?

Câu hỏi lab đặt ra là **khi nào orchestration đáng giá**. Muốn trả lời được, baseline
phải nhận **cùng evidence** như crew — nếu không, phép đo trở thành đo retrieval.

Đo thật trên workload này: multi-agent **không** thắng. Quality 8.2 vs 8.2, citation
coverage 73% vs 73%, nhưng tốn 2× latency và 2.1× cost. Lý do: retrieval ở đây quá dễ
(corpus trả đúng 5 nguồn ngay lần đầu) và task chỉ là "đọc 5 tài liệu rồi tổng hợp" —
một model đủ tốt làm xong trong một lượt. Handoff sang analyst, vòng critic, viết lại
đều là coordination overhead không có việc để biện minh.

Điều đó **không** có nghĩa multi-agent vô dụng, mà là workload này chọn sai để chứng minh.
Nơi kiến trúc này đáng giá là khi task cần nhiều nguồn thông tin khác nhau, cần nhiều vòng
truy vấn, hoặc cần verification độc lập — xem mục Exit ticket.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Chọn route kế tiếp, enforce guardrail | Toàn bộ state | `route_history` | Loop vô hạn nếu thiếu iteration guard |
| Researcher | Retrieve + cô đọng evidence | query, `max_sources` | `sources`, `research_notes` | Search fail; corpus thiếu topic khớp |
| Analyst | Cấu trúc hóa thành 5 mục insight | `research_notes`, fact bank | `analysis_notes` | Bịa claim ngoài notes |
| Writer | Tổng hợp báo cáo có citation | notes + analysis | `final_answer` | Bịa `source_id` không tồn tại |
| Critic | Verify citation (deterministic) | `final_answer`, `sources` | Coverage report | Không bắt được lỗi ngữ nghĩa, chỉ bắt lỗi id |

## Shared state

`ResearchState` (`core/state.py`) — mọi agent đọc/ghi cùng một object:

| Field | Vì sao cần |
|---|---|
| `request` | Query + `max_sources` + audience, bất biến suốt run |
| `iteration`, `route_history` | Guardrail chống loop; đồng thời là trace giải thích "ai làm gì" |
| `sources` | Evidence gốc kèm `source_id`, `is_synthetic`, `full_text` trong metadata |
| `research_notes` → `analysis_notes` → `final_answer` | Ba mốc handoff; supervisor route dựa trên field nào còn `None` |
| `agent_results` | Output + metadata từng agent, phục vụ benchmark |
| `trace` | Event log có cấu trúc, export JSON |
| `errors` | Prefix `"{agent}: ..."` để supervisor đếm failure per-agent |

Thiết kế quan trọng: **routing đọc được từ state**, không cần LLM quyết định — rẻ, tất định, dễ giải thích.

## Routing policy

`SupervisorAgent.decide()` (`agents/supervisor.py`), theo thứ tự phụ thuộc:

```text
iteration >= max_iterations         -> done      (guardrail thắng mọi điều kiện khác)
chưa có sources                     -> researcher
có sources, chưa có analysis_notes  -> analyst
có analysis, chưa có final_answer   -> writer
đã có answer, critic chưa chạy      -> critic
còn lại                             -> done
```

Mỗi nhánh còn kiểm tra `_blocked()`: một agent fail >= 2 lần thì bị **skip** thay vì retry
mãi — pipeline đi tiếp với thông tin thiếu, tốt hơn là treo.

## Guardrails

- **Max iterations:** `MAX_ITERATIONS=6` (env), check đầu tiên trong `decide()`;
  LangGraph còn có `recursion_limit` riêng.
- **Timeout:** `TIMEOUT_SECONDS=60` truyền vào OpenAI client.
- **Retry:** `tenacity`, 3 lần, exponential backoff 1-8s, chỉ ở tầng `LLMClient`.
- **Fallback:** mỗi worker có `_fallback_*()` deterministic — LLM chết thì pipeline vẫn
  ra kết quả (giảm chất lượng) thay vì crash.
- **Validation:** Pydantic ở mọi I/O; Critic verify citation; `count_hallucinated_citations()`
  phạt điểm quality.

## Tracing

`observability/providers.py` cài **OpenTelemetry** làm provider mặc định, chọn nó thay vì
LangSmith/Langfuse vì OTel chạy được **không cần API key** — bài nộp verify được ngay trên
máy chấm. LangSmith tự động bật nếu `LANGSMITH_API_KEY` có trong `.env`. Provider hỏng
không bao giờ làm gãy workflow (`trace_span` nuốt exception của provider).

```bash
python -m multi_agent_research_lab.cli multi-agent -q "..." --tracer otel
# -> tracing provider: opentelemetry / opentelemetry: exported 5 spans
```

Song song đó `export_trace_json()` ghi trace tự chứa ra `reports/trace_demo.json`, và
`--screenshot` render trace thành PNG (`reports/screenshots/`) — sinh từ dữ liệu trace
nên tái tạo được, không phải ảnh chụp màn hình thủ công.

Một guardrail phát hiện khi làm bài: prompt ban đầu dùng `[synthetic]` làm nhãn cảnh báo,
trùng cú pháp với citation `[source_id]` khiến model nhét nhãn vào thành citation giả.
Đã sửa: **dấu ngoặc vuông chỉ dành cho citation id**, nhãn dùng `(ngoặc tròn)`.

## Benchmark plan

3 query trong `configs/lab_default.yaml`, mỗi query chạy 2 chế độ.

| Metric | Cách đo |
|---|---|
| Latency | `perf_counter()` quanh runner |
| Cost | Token thật từ `usage` × bảng giá trong `llm_client.py` |
| Quality (0-10) | Heuristic: substance 0-3 + grounding 0-4 + structure 0-2 − phạt hallucination/errors |
| Citation coverage | `|cited ∩ valid| / |valid|` |
| Failure rate | Run ném exception / tổng run |

Quality score là **sàng lọc tự động**, không thay peer review rubric — nó thưởng grounding
và cấu trúc, những thứ đếm được.

## Failure modes gặp thật và cách fix

Bốn lỗi dưới đây phát hiện khi test, không phải giả định.

### 1. Nhãn metadata bị nuốt thành citation giả

**Triệu chứng:** Critic báo `hallucinated citations: M001, source_id` dù model không bịa gì.

**Nguyên nhân:** prompt dùng `[synthetic]` làm nhãn cảnh báo, trùng cú pháp với citation
`[source_id]`. Agent thấy `[...]` là coi như citation.

**Fix:** quy ước lại — **dấu ngoặc vuông chỉ dành cho citation id**, nhãn dùng `(ngoặc tròn)`.
Prompt nói rõ "Square brackets are RESERVED for citation ids". Đây là lỗi prompt design,
LLM thật cũng mắc chứ không riêng mock.

### 2. Regex citation bỏ sót id ngắn

**Triệu chứng:** citation coverage luôn 0% dù bài viết có trích dẫn.

**Nguyên nhân:** regex `[A-Za-z0-9_\-]{2,40}` yêu cầu tối thiểu 2 ký tự, nên `[a]` bị bỏ qua.

**Fix:** đổi thành `{1,40}`. Unit test `test_citation_coverage_counts_only_valid_ids` chốt lại.

### 3. Key trong `.env` bị biến môi trường global che mất

**Triệu chứng:** 401 `invalid_api_key` dù key trong `.env` hoàn toàn hợp lệ.

**Nguyên nhân:** cả `load_dotenv()` lẫn pydantic-settings mặc định **ưu tiên biến môi trường
hơn file `.env`**. Một `OPENAI_API_KEY` cũ còn sót trong shell sẽ âm thầm thắng.

**Fix:** override `settings_customise_sources` trong `core/config.py` để `.env` của repo
thắng environment. Lỗi này nguy hiểm vì thông báo 401 trông y hệt như key sai thật.

### 4. Citation vô hình trên terminal

**Triệu chứng:** bài viết in ra thiếu hẳn citation, nhưng file trace lại có đủ.

**Nguyên nhân:** Rich hiểu `[autogen]` là markup tag và nuốt mất khi render.

**Fix:** bọc output của model bằng `rich.text.Text()` để tắt markup parsing.

### 5. Baseline không retrieval làm sai lệch cả kết luận

**Triệu chứng:** bản benchmark đầu cho multi-agent thắng áp đảo — citation coverage
0% vs 87%, quality 4.0 vs 6.5.

**Nguyên nhân:** baseline khi đó **không có bước retrieval nào**. Một bài viết không được
cấp nguồn thì coverage 0% là tất yếu, không liên quan gì đến orchestration. Phép đo đang
đo *retrieval*, trong khi câu hỏi lab hỏi về *orchestration*.

**Fix:** baseline giờ nhận đúng evidence như crew, chỉ khác là làm mọi việc trong 1 LLM
call. Kết quả đảo ngược thành "multi-agent không thắng" — và đó mới là câu trả lời trung
thực. Arm cũ giữ lại dưới tên `baseline-noretrieval` để thấy rõ phép so sánh cũ sai thế nào.

**Bài học:** một kết quả đẹp bất thường thường là dấu hiệu control arm bị thiết kế yếu,
không phải dấu hiệu hệ thống tốt.

### Failure mode còn tồn tại (chưa fix)

- **Critic chỉ kiểm tra id, không kiểm tra ngữ nghĩa.** Một câu trích `[autogen]` nhưng
  nội dung sai hoàn toàn so với nguồn vẫn PASS. Muốn bắt được cần entailment check —
  chi phí một LLM call nữa, và lại quay về vấn đề model tự chấm.
- **LLM-as-judge không phân biệt được các arm.** Judge (`evaluation/judge.py`) thay heuristic
  cũ, nhưng với `gpt-4o-mini` nó chấm gần như mọi bài 7.5-8.5 — kể cả bài không có citation
  nào. Nó không đủ độ phân giải để tách chất lượng giữa hai arm. Judge cũng chạy cùng model
  family với agent nên chỉ là tín hiệu **tương đối**; peer review vẫn là ground truth.
  Heuristic `score_quality()` giữ lại làm fallback khi judge không khả dụng.

## Exit ticket

**Nên dùng multi-agent khi:** task cần **nhiều loại thông tin khác nhau** hoặc **nhiều vòng
truy vấn** (retrieval khó, phải thử lại, phải bắc cầu giữa nhiều nguồn), hoặc khi cần
**verification độc lập** mà một model không tự làm đáng tin được. Trên workload này Critic
vẫn có giá trị riêng: nó bắt citation bịa bằng string matching, không phụ thuộc model.

**Không nên khi:** retrieval dễ và task gói gọn trong một lượt — chính là ca đo được ở đây.
Multi-agent tốn 2× latency, 2.1× cost, thêm bề mặt lỗi ở mỗi handoff, mà quality và citation
coverage đều **ngang** baseline. Đây là kết quả âm, và nó là câu trả lời trung thực cho
câu hỏi lab chứ không phải dấu hiệu implementation hỏng.
