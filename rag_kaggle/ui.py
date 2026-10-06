from __future__ import annotations

import json

from .pipeline import RAGPipeline


def build_demo(pipeline: RAGPipeline):
    import gradio as gr

    def ingest_files(files):
        if not files:
            return "Chưa có file.", None
        paths = [file.name if hasattr(file, "name") else str(file) for file in files]
        try:
            report = pipeline.ingest(paths, reset=True)
            archive = pipeline.export_artifacts()
            return json.dumps(report, ensure_ascii=False, indent=2), str(archive)
        except Exception as exc:
            return f"Ingestion failed: {exc}", None

    def chat(message, history):
        try:
            result = pipeline.ask(message)
        except Exception as exc:
            return f"Pipeline error: {exc}"
        citations = "\n".join(f"- {citation}" for citation in result["citations"])
        trace = "\n".join(
            f"- `{item['chunk_id']}` | {item['chunk_type']} | "
            f"RRF={item['rrf_score']:.4f} | rerank={item['rerank_score']}"
            for item in result["trace"]
        )
        response = result["answer"]
        if citations:
            response += "\n\n**Nguồn**\n" + citations
        if trace:
            response += "\n\n<details><summary>Retrieval trace</summary>\n\n" + trace + "\n</details>"
        return response

    with gr.Blocks(title="Multimodal RAG - Kaggle") as demo:
        gr.Markdown(
            "# Multimodal RAG\n"
            "PDF, DOCX, Excel · PaddleOCR-VL · Parent–Child · Qdrant · BM25 · RRF · Qwen"
        )
        with gr.Tab("1. Ingestion"):
            files = gr.File(
                label="Upload tài liệu",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".xlsx", ".xlsm", ".xls"],
                type="filepath",
            )
            ingest_button = gr.Button("Parse và lập chỉ mục", variant="primary")
            ingest_report = gr.Code(label="Ingestion report", language="json")
            artifact = gr.File(label="Tải artifacts")
            ingest_button.click(ingest_files, inputs=files, outputs=[ingest_report, artifact])

        with gr.Tab("2. Chat"):
            gr.ChatInterface(fn=chat, type="messages")

        with gr.Tab("3. System"):
            gr.JSON(value=pipeline.metadata.stats(), label="Database stats")

    return demo
