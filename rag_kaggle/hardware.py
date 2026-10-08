"""GPU placement helpers for Kaggle's variable accelerator inventory."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .config import PipelineConfig


def configure_kaggle_devices(config: PipelineConfig, gpu_count: int | None = None) -> dict[str, Any]:
    """Pin independent models to different GPUs when Kaggle provides T4 x2.

    Qwen 7B in 4-bit fits on one T4, so tensor-parallel sharding is counterproductive
    for this interactive workload. GPU 0 is reserved for generation/VLM. GPU 1 is
    shared sequentially by PaddleOCR-VL during parsing and BGE during embedding;
    the pipeline terminates the OCR worker before it starts dense indexing. The
    lower-volume reranking stage runs on CPU to use Kaggle's available system RAM.
    """
    if gpu_count is None:
        try:
            import torch

            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except ImportError:
            gpu_count = 0

    if gpu_count >= 2:
        config.parsing.ocr_cuda_visible_devices = "1"
        config.retrieval.dense_device = "cuda:1"
        config.retrieval.reranker_device = "cpu"
        config.retrieval.reranker_use_fp16 = False
        config.generation.device = "cuda:0"
        config.vision.device = "cuda:0"
        return {
            "mode": "dual_t4",
            "gpu_0": ["Qwen answer", "optional Qwen-VL"],
            "gpu_1": ["PaddleOCR-VL worker", "BGE dense embedding"],
            "cpu": ["BGE reranker", "BM25", "Qdrant", "parsing/chunking", "structured computation"],
        }

    if gpu_count == 1:
        config.parsing.ocr_cuda_visible_devices = "0"
        config.retrieval.dense_device = "cuda:0"
        config.retrieval.reranker_device = "cuda:0"
        config.generation.device = "cuda:0"
        config.vision.device = "cuda:0"
        return {
            "mode": "single_gpu",
            "gpu_0": ["PaddleOCR-VL", "embedding", "reranker", "Qwen", "optional Qwen-VL"],
        }

    config.retrieval.dense_device = "cpu"
    config.retrieval.reranker_device = "cpu"
    config.retrieval.reranker_use_fp16 = False
    config.generation.device = None
    config.vision.device = None
    return {"mode": "cpu_only", "warning": "No CUDA GPU was detected."}


def configure_ingestion_devices(config: PipelineConfig, gpu_count: int | None = None) -> dict[str, Any]:
    """Allocate accelerators without reserving memory for an answer model."""
    gpu_count = _gpu_count(gpu_count)
    if gpu_count >= 2:
        config.parsing.ocr_cuda_visible_devices = "1"
        config.retrieval.dense_device = "cuda:0"
        config.vision.device = "cuda:0"
        # GPU 0 is idle while pages are parsed: host the OCR-correction LLM there. It is
        # unloaded before dense embedding starts, so the two never compete for memory.
        config.correction.device = "cuda:0"
        return {
            "mode": "ingestion_dual_gpu",
            "gpu_0": ["OCR correction LLM (during parsing)", "optional vision model", "dense embedding (after parsing)"],
            "gpu_1": ["PaddleOCR-VL worker"],
            "cpu": ["parsing", "chunking", "Qdrant", "BM25", "metadata"],
        }
    if gpu_count == 1:
        config.parsing.ocr_cuda_visible_devices = "0"
        config.retrieval.dense_device = "cuda:0"
        config.vision.device = "cuda:0"
        config.correction.device = "cuda:0"
        return {
            "mode": "ingestion_single_gpu",
            "gpu_0": ["OCR", "OCR correction LLM", "vision", "embedding (sequential)"],
            "warning": "OCR and the correction LLM share one GPU; disable correction if memory runs out.",
        }
    config.retrieval.dense_device = "cpu"
    config.vision.device = None
    config.correction.enabled = False  # A 7B model on CPU is impractical.
    return {
        "mode": "ingestion_cpu_only",
        "warning": "No CUDA GPU was detected; OCR correction was disabled.",
    }


def configure_retrieval_devices(config: PipelineConfig, gpu_count: int | None = None) -> dict[str, Any]:
    """Allocate only query embedding, reranking and answer-generation resources."""
    gpu_count = _gpu_count(gpu_count)
    if gpu_count >= 2:
        config.retrieval.dense_device = "cuda:1"
        config.retrieval.reranker_device = "cpu"
        config.retrieval.reranker_use_fp16 = False
        config.generation.device = "cuda:0"
        return {
            "mode": "retrieval_dual_gpu",
            "gpu_0": ["answer/query-rewrite model"],
            "gpu_1": ["query embedding"],
            "cpu": ["reranker", "Qdrant", "BM25", "context expansion"],
        }
    if gpu_count == 1:
        config.retrieval.dense_device = "cuda:0"
        config.retrieval.reranker_device = "cuda:0"
        config.generation.device = "cuda:0"
        return {"mode": "retrieval_single_gpu", "gpu_0": ["embedding", "reranker", "answer model"]}
    config.retrieval.dense_device = "cpu"
    config.retrieval.reranker_device = "cpu"
    config.retrieval.reranker_use_fp16 = False
    config.generation.device = None
    return {"mode": "retrieval_cpu_only", "warning": "No CUDA GPU was detected."}


def inspect_resources(path: str | Path = "/kaggle/working") -> dict[str, Any]:
    """Return reproducible hardware facts used when choosing larger models."""
    report: dict[str, Any] = {"gpus": []}
    try:
        import torch

        for index in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
            props = torch.cuda.get_device_properties(index)
            free_bytes, _ = torch.cuda.mem_get_info(index)
            report["gpus"].append(
                {
                    "index": index,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / 1024**3, 2),
                    "free_vram_gb": round(free_bytes / 1024**3, 2),
                }
            )
    except Exception:
        pass
    try:
        import psutil

        memory = psutil.virtual_memory()
        report["ram_gb"] = round(memory.total / 1024**3, 2)
        report["ram_available_gb"] = round(memory.available / 1024**3, 2)
    except Exception:
        report["ram_gb"] = report["ram_available_gb"] = None
    disk_path = Path(path)
    if not disk_path.exists():
        disk_path = Path.cwd()
    disk = shutil.disk_usage(disk_path)
    report["disk_free_gb"] = round(disk.free / 1024**3, 2)
    return report


def suggest_model_upgrades(stage: str, resources: dict[str, Any], config: PipelineConfig) -> list[str]:
    """Return conservative suggestions; never changes a model automatically."""
    gpus = resources.get("gpus") or []
    maximum_vram = max((gpu.get("vram_gb", 0) for gpu in gpus), default=0)
    suggestions: list[str] = []
    if stage == "ingestion" and maximum_vram >= 24 and config.retrieval.dense_model == "BAAI/bge-m3":
        suggestions.append(
            "Có GPU >=24 GB: có thể benchmark BAAI/bge-multilingual-gemma2; đổi embedding bắt buộc ingest lại."
        )
    if stage == "retrieval" and maximum_vram >= 24 and "7B" in config.generation.model:
        suggestions.append(
            "Có GPU >=24 GB: có thể benchmark Qwen/Qwen2.5-14B-Instruct 4-bit cho answer quality."
        )
    if not suggestions:
        suggestions.append("Chưa thấy headroom đủ an toàn để tự đề xuất model lớn hơn; giữ cấu hình hiện tại.")
    return suggestions


def _gpu_count(value: int | None) -> int:
    if value is not None:
        return value
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        return 0
