from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from .pipeline import RAGPipeline


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
HIT_HEADERS = ["#", "file", "type", "page/sheet", "dense", "bm25", "RRF", "rerank", "preview"]
ALL = "(tất cả)"


def build_demo(pipeline: RAGPipeline):
    import gradio as gr

    def ingest_files(files, reset):
        if not files:
            yield "Chưa có file.", "", None, _documents(pipeline)
            return
        paths = [file.name if hasattr(file, "name") else str(file) for file in files]
        messages: queue.Queue = queue.Queue()
        outcome: dict = {}

        def worker():
            try:
                outcome["report"] = pipeline.ingest(paths, reset=reset, progress=messages.put)
                outcome["archive"] = str(pipeline.export_artifacts())
            except Exception as exc:  # Reported in the UI instead of crashing the demo.
                outcome["error"] = str(exc)
            finally:
                messages.put(None)

        threading.Thread(target=worker, daemon=True).start()
        log_lines = []
        while True:
            message = messages.get()
            if message is None:
                break
            log_lines.append(message)
            yield "\n".join(log_lines), "", None, _documents(pipeline)
        if "error" in outcome:
            log_lines.append(f"Ingestion failed: {outcome['error']}")
            yield "\n".join(log_lines), "", None, _documents(pipeline)
            return
        report = {key: value for key, value in outcome["report"].items() if key != "statuses"}
        yield (
            "\n".join(log_lines),
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            outcome.get("archive"),
            _documents(pipeline),
        )

    def respond(message, history, source_file, content_type):
        history = list(history or [])
        if not message or not message.strip():
            return history, "", [], [], {}, ""
        filters = {
            "source_file": None if source_file in (None, ALL) else source_file,
            "chunk_type": None if content_type in (None, ALL) else content_type,
        }
        try:
            result = pipeline.ask(message, filters=filters)
        except Exception as exc:
            history += [{"role": "user", "content": message}, {"role": "assistant", "content": f"Pipeline error: {exc}"}]
            return history, "", [], [], {}, ""

        answer = result["answer"]
        if result.get("citations"):
            answer += "\n\n**Nguồn**\n" + "\n".join(f"- {citation}" for citation in result["citations"])
        if result.get("warnings"):
            answer += "\n\n> ⚠️ " + "\n> ⚠️ ".join(result["warnings"])
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]

        rows = [
            [
                index,
                item["source_file"],
                item["chunk_type"],
                item["page"] if item["page"] is not None else f"{item['sheet_name'] or ''} {item['cell_range'] or ''}".strip(),
                item["dense_rank"],
                item["sparse_rank"],
                round(item["rrf_score"], 4),
                None if item["rerank_score"] is None else round(item["rerank_score"], 4),
                item["preview"][:300],
            ]
            for index, item in enumerate(result.get("trace", []), start=1)
        ]
        images = []
        for context in result.get("contexts", []):
            for path in context.get("asset_paths", []):
                if Path(path).suffix.lower() in IMAGE_SUFFIXES and Path(path).exists():
                    images.append((path, context["citation"]))
        contexts_md = "\n\n".join(
            f"**{context['source_id']}** {context['citation']}\n```text\n{context['preview']}\n```"
            for context in result.get("contexts", [])
        )
        trace = {
            key: result.get(key)
            for key in ("trace_id", "query_plan", "filters", "computation", "latency_ms", "refusal_reason", "source_ids")
        }
        return history, "", rows, images, trace, contexts_md

    def run_evaluation(dataset, run_answers):
        if not dataset:
            return {"error": "Chưa chọn file dataset JSONL."}
        path = dataset.name if hasattr(dataset, "name") else str(dataset)
        try:
            report = pipeline.evaluate(path, run_answers=run_answers)
        except Exception as exc:
            return {"error": str(exc)}
        return {key: value for key, value in report.items() if key != "rows"}

    def restore(archive):
        if not archive:
            return {"error": "Chưa chọn file zip."}, _documents(pipeline)
        path = archive.name if hasattr(archive, "name") else str(archive)
        try:
            return pipeline.restore_artifacts(path), _documents(pipeline)
        except Exception as exc:
            return {"error": str(exc)}, _documents(pipeline)

    def filter_choices():
        files = [ALL, *pipeline.metadata.list_values("source_file")]
        types = [ALL, *pipeline.metadata.list_values("chunk_type")]
        return gr.update(choices=files, value=ALL), gr.update(choices=types, value=ALL)

    with gr.Blocks(title="Multimodal RAG - Kaggle") as demo:
        gr.Markdown(
            "# Multimodal RAG\n"
            "PDF, DOCX, Excel · PaddleOCR-VL · Parent–Child · Qdrant · BM25 · RRF · Reranker · Qwen"
        )
        with gr.Tab("1. Ingestion"):
            files = gr.File(
                label="Upload tài liệu",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".xlsx", ".xlsm", ".xls"],
                type="filepath",
            )
            reset = gr.Checkbox(label="Xoá toàn bộ index cũ trước khi ingest", value=False)
            ingest_button = gr.Button("Parse và lập chỉ mục", variant="primary")
            stage_log = gr.Textbox(label="Trạng thái từng stage", lines=12, max_lines=30)
            ingest_report = gr.Code(label="Ingestion report", language="json")
            artifact = gr.File(label="Tải artifacts (Qdrant + SQLite + parsed + assets)")

        with gr.Tab("2. Chat"):
            with gr.Row():
                source_filter = gr.Dropdown(label="Lọc theo file", choices=[ALL], value=ALL)
                type_filter = gr.Dropdown(label="Lọc theo loại nội dung", choices=[ALL], value=ALL)
                refresh_filters = gr.Button("Làm mới bộ lọc")
            chatbot = _chatbot(gr)
            question = gr.Textbox(label="Câu hỏi", placeholder="Ví dụ: Phí thường niên thẻ Visa Gold là bao nhiêu?")
            ask_button = gr.Button("Hỏi", variant="primary")
            with gr.Accordion("Retrieved child chunks và điểm số", open=False):
                hits_table = gr.Dataframe(headers=HIT_HEADERS, wrap=True)
            with gr.Accordion("Context gửi cho LLM", open=False):
                contexts_view = gr.Markdown()
            with gr.Accordion("Preview nguồn (ảnh trang/hình)", open=False):
                gallery = gr.Gallery(columns=3, height=360)
            with gr.Accordion("Trace", open=False):
                trace_view = gr.JSON()

        with gr.Tab("3. Evaluation"):
            gr.Markdown(
                "Upload file JSONL: mỗi dòng có `question`, `expected_answer`, `expected_document`, "
                "`expected_location`, `expected_block_ids`, `content_type`, `category`."
            )
            dataset = gr.File(label="Dataset JSONL", file_types=[".jsonl", ".json"], type="filepath")
            run_answers = gr.Checkbox(label="Chạy cả answer generation (chậm hơn)", value=True)
            evaluate_button = gr.Button("Chạy evaluation", variant="primary")
            evaluation_report = gr.JSON(label="Metrics")

        with gr.Tab("4. System"):
            refresh = gr.Button("Làm mới")
            stats = gr.JSON(value=pipeline.metadata.stats(), label="Database stats")
            documents = gr.Dataframe(value=_documents(pipeline), headers=["document_id", "source_file", "file_type"])
            statuses = gr.Dataframe(value=_statuses(pipeline), label="Ingestion status (mới nhất trước)")
            restore_file = gr.File(label="Khôi phục từ rag_artifacts.zip", file_types=[".zip"], type="filepath")
            restore_button = gr.Button("Khôi phục artifacts")

        ingest_button.click(
            ingest_files, inputs=[files, reset], outputs=[stage_log, ingest_report, artifact, documents]
        ).then(filter_choices, outputs=[source_filter, type_filter])
        chat_outputs = [chatbot, question, hits_table, gallery, trace_view, contexts_view]
        chat_inputs = [question, chatbot, source_filter, type_filter]
        ask_button.click(respond, inputs=chat_inputs, outputs=chat_outputs)
        question.submit(respond, inputs=chat_inputs, outputs=chat_outputs)
        refresh_filters.click(filter_choices, outputs=[source_filter, type_filter])
        evaluate_button.click(run_evaluation, inputs=[dataset, run_answers], outputs=evaluation_report)
        refresh.click(
            lambda: (pipeline.metadata.stats(), _documents(pipeline), _statuses(pipeline)),
            outputs=[stats, documents, statuses],
        )
        restore_button.click(restore, inputs=restore_file, outputs=[stats, documents]).then(
            filter_choices, outputs=[source_filter, type_filter]
        )
        demo.load(filter_choices, outputs=[source_filter, type_filter])

    return demo


def launch_demo(pipeline: RAGPipeline, **kwargs):
    """Launch with access to runtime assets and the artifact archive."""
    demo = build_demo(pipeline)
    allowed = {str(pipeline.config.work_dir), str(pipeline.config.work_dir.parent)}
    kwargs.setdefault("allowed_paths", sorted(allowed))
    return demo.queue(default_concurrency_limit=1).launch(**kwargs)


def _chatbot(gr):
    try:
        return gr.Chatbot(type="messages", height=480, label="Hội thoại")
    except TypeError:  # Gradio 6 uses the messages format only.
        return gr.Chatbot(height=480, label="Hội thoại")


def _documents(pipeline: RAGPipeline) -> list[list[str]]:
    return [[row["document_id"], row["source_file"], row["file_type"]] for row in pipeline.metadata.list_documents()]


def _statuses(pipeline: RAGPipeline) -> list[list]:
    return [
        [row["created_at"], row["source_file"], row["stage"], row["status"], row["error_code"], row["message"]]
        for row in pipeline.metadata.statuses(limit=100)
    ]
