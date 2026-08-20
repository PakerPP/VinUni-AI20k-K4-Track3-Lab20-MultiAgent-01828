# Trace Screenshots

Deliverable #2 của lab: *"Screenshot trace hoặc link trace"*.

## Ảnh chụp thủ công (chạy thật với `gpt-4o-mini`)

| Ảnh | Nội dung | Chứng minh điều gì |
|---|---|---|
| [`run_logs_and_answer.png`](run_logs_and_answer.png) | Lệnh chạy + log từng bước + đầu bài viết | Chạy thật với OpenAI API (`HTTP/1.1 200 OK`), kèm `in_tokens` / `out_tokens` / `cost_usd` cho từng agent |
| [`answer_with_citations.png`](answer_with_citations.png) | Phần còn lại của bài viết | Citation `[autogen]`, `[metagpt]`, `[agentbench]`, `[anthropic_agents]` trích dẫn inline từ corpus offline |
| [`critic_and_trace.png`](critic_and_trace.png) | Critic Review + bảng Trace 10 bước | Citation coverage 100%, 0 citation bịa; trace cho thấy supervisor điều phối từng bước |
| [`trace_second_run.png`](trace_second_run.png) | Bảng Trace của một run độc lập khác | Routing ổn định qua nhiều lần chạy (`notes_chars` 2494 vs 2286 — nội dung khác, đường đi giống) |

Lệnh tái tạo:

```bash
python -m multi_agent_research_lab.cli multi-agent \
  -q "When is a multi-agent architecture better than a single agent for research tasks?" \
  --max-sources 4
```

## Render trace thành ảnh bằng code

Ngoài ảnh chụp tay, CLI có thể tự vẽ trace thành PNG từ chính dữ liệu trace
(`observability/screenshot.py`) — tái tạo được, không phụ thuộc thao tác chụp:

```bash
python -m multi_agent_research_lab.cli multi-agent -q "..."   --screenshot reports/screenshots/trace.png
```

Ảnh mẫu không commit vào repo; chạy lệnh trên là sinh ra.

## Trace đầy đủ dạng dữ liệu

[`../trace_demo.json`](../trace_demo.json) — spans, event, routing, và `cost_summary`
(token + cost + wall-time theo từng agent).
