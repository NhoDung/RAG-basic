"""GPU placement helpers for Kaggle's variable accelerator inventory."""

from __future__ import annotations

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
