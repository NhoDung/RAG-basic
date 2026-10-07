"""Kaggle-oriented multimodal RAG baseline."""

from .config import PipelineConfig
from .hardware import (
    configure_ingestion_devices,
    configure_kaggle_devices,
    configure_retrieval_devices,
    inspect_resources,
    suggest_model_upgrades,
)
from .pipeline import IngestionPipeline, RAGPipeline, RetrievalAnswerPipeline

__all__ = [
    "IngestionPipeline",
    "PipelineConfig",
    "RAGPipeline",
    "RetrievalAnswerPipeline",
    "configure_ingestion_devices",
    "configure_kaggle_devices",
    "configure_retrieval_devices",
    "inspect_resources",
    "suggest_model_upgrades",
]
