from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


class PaddleOCRVLAdapter:
    """Small compatibility layer around PaddleOCR-VL releases.

    PaddleOCR has changed constructor and result field names between releases.
    This adapter keeps those differences out of the document parsers and preserves
    the raw result for debugging.
    """

    def __init__(self, model_name: str, model_dir: str | None = None):
        self.model_name = model_name
        self.model_dir = model_dir
        self.pipeline = None

    def load(self) -> None:
        if self.pipeline is not None:
            return
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR-VL is not installed. Run the dependency cell in the Kaggle notebook."
            ) from exc

        attempts: list[dict[str, Any]] = []
        if self.model_dir:
            attempts.append({"vl_rec_model_dir": self.model_dir})
            attempts.append({"model_dir": self.model_dir})
        attempts.append({"vl_rec_model_name": self.model_name})
        attempts.append({"model_name": self.model_name})
        attempts.append({})

        errors = []
        for kwargs in attempts:
            try:
                self.pipeline = PaddleOCRVL(**kwargs)
                if not kwargs:
                    LOGGER.warning(
                        "PaddleOCRVL did not accept an explicit model selector; using the package default. "
                        "Pin the PaddleOCR package/model dataset to guarantee version %s.",
                        self.model_name,
                    )
                return
            except (TypeError, ValueError) as exc:
                errors.append(f"{kwargs}: {exc}")

        raise RuntimeError("Unable to initialize PaddleOCR-VL:\n" + "\n".join(errors))

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        self.load()
        results = list(self.pipeline.predict(str(image_path)))
        raw = [self._to_serializable(result) for result in results]
        blocks = []
        for item in raw:
            blocks.extend(self._collect_layout_blocks(item))
        if blocks:
            text = "\n\n".join(block["content"] for block in blocks if block["content"])
        else:
            text_parts = []
            for item in raw:
                text_parts.extend(self._collect_text(item))
            text = "\n".join(dict.fromkeys(part.strip() for part in text_parts if part.strip()))
        return {"text": text, "blocks": blocks, "raw": raw, "model": self.model_name}

    def unload(self) -> None:
        self.pipeline = None
        try:
            import gc
            import paddle
            import torch

            gc.collect()
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _to_serializable(self, value: Any) -> Any:
        for attribute in ("json", "res", "result"):
            candidate = getattr(value, attribute, None)
            if callable(candidate):
                try:
                    candidate = candidate()
                except TypeError:
                    continue
            if candidate is not None:
                if isinstance(candidate, str):
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        return {"text": candidate}
                return self._normalize(candidate)
        if hasattr(value, "to_dict"):
            return self._normalize(value.to_dict())
        return self._normalize(value)

    def _normalize(self, value: Any) -> Any:
        if hasattr(value, "to_dict"):
            try:
                return self._normalize(value.to_dict())
            except TypeError:
                pass
        if isinstance(value, dict):
            return {str(key): self._normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._normalize(item) for item in value]
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _collect_layout_blocks(self, value: Any) -> list[dict[str, Any]]:
        """Extract layout blocks (label, content, bbox, order) in reading order.

        PaddleOCR-VL exposes them as ``parsing_res_list`` entries with keys such as
        ``block_label``/``block_content``/``block_bbox``; field names differ between
        releases, so several aliases are accepted.
        """
        label_keys = ("block_label", "label", "type")
        content_keys = ("block_content", "content", "text", "markdown", "html")
        bbox_keys = ("block_bbox", "bbox", "coordinate", "box")
        blocks: list[dict[str, Any]] = []

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                label = next((node[key] for key in label_keys if isinstance(node.get(key), str)), None)
                content = next((node[key] for key in content_keys if isinstance(node.get(key), str)), None)
                if label and content and content.strip():
                    bbox = next((node[key] for key in bbox_keys if isinstance(node.get(key), list)), None)
                    score = node.get("score", node.get("confidence"))
                    blocks.append(
                        {
                            "label": label.lower(),
                            "content": content.strip(),
                            "bbox": bbox if bbox and all(isinstance(v, (int, float)) for v in bbox) else None,
                            "confidence": float(score) if isinstance(score, (int, float)) else None,
                        }
                    )
                    return
                for item in node.values():
                    visit(item)
            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(value)
        return blocks

    def _collect_text(self, value: Any) -> list[str]:
        text_keys = {
            "text",
            "texts",
            "rec_text",
            "rec_texts",
            "markdown",
            "content",
            "block_content",
            "block_text",
            "label",
            "html",
        }
        collected: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in text_keys:
                    if isinstance(item, str):
                        collected.append(item)
                    elif isinstance(item, list):
                        collected.extend(str(part) for part in item if isinstance(part, (str, int, float)))
                collected.extend(self._collect_text(item))
        elif isinstance(value, list):
            for item in value:
                collected.extend(self._collect_text(item))
        return collected
