from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from .config import PipelineConfig


LOGGER = logging.getLogger(__name__)

FLOWCHART_HINTS = ("sơ đồ", "lưu đồ", "quy trình", "flowchart", "flow chart", "workflow", "diagram", "process")
CHART_HINTS = ("biểu đồ", "đồ thị", "chart", "graph", "dashboard")


def classify_image(caption: str = "", ocr_text: str = "", ocr_labels: list[str] | None = None) -> str:
    """Cheap image classification from caption, OCR text and OCR layout labels."""
    labels = " ".join(ocr_labels or []).lower()
    caption = (caption or "").lower()
    if "chart" in labels or any(hint in caption for hint in CHART_HINTS):
        return "chart"
    if any(hint in caption for hint in FLOWCHART_HINTS):
        return "flowchart"
    text = (ocr_text or "").lower()
    arrows = len(re.findall(r"→|->|=>|⟶|➔", text))
    if arrows >= 2 or any(hint in text for hint in ("bắt đầu", "kết thúc", "start", "end")) and "\n" in text:
        return "flowchart"
    return "image"


FLOWCHART_PROMPT = """Ảnh là một sơ đồ/flowchart trong tài liệu nghiệp vụ tiếng Việt.
Tiêu đề/caption: {caption}
Mục: {section}
{previous}Văn bản OCR (PaddleOCR-VL, đã sửa chính tả, dùng làm nguồn chữ chính xác):
{ocr_text}

Hãy mô tả logic của sơ đồ. Chỉ dùng nhãn có trong ảnh/OCR, không bịa thêm bước.
Trả về DUY NHẤT một JSON object theo schema:
{{"image_type": "flowchart", "title": str, "summary": str,
  "nodes": [{{"id": str, "label": str, "type": "start|process|decision|end|other"}}],
  "edges": [{{"from": str, "to": str, "condition": str|null}}],
  "ocr_text": [str], "uncertainties": [str]}}"""

CHART_PROMPT = """Ảnh là một biểu đồ/dashboard trong tài liệu nghiệp vụ tiếng Việt.
Tiêu đề/caption: {caption}
Mục: {section}
{previous}Văn bản OCR (PaddleOCR-VL, đã sửa chính tả):
{ocr_text}

Tách riêng quan sát lấy trực tiếp từ nhãn/số liệu nhìn thấy (observations) và nhận xét
do bạn suy luận (summary). Không tạo số liệu không có trong ảnh.
Trả về DUY NHẤT một JSON object theo schema:
{{"image_type": "chart", "title": str, "chart_type": str, "axes": [str], "legend": [str],
  "observations": [str], "summary": str, "uncertainties": [str]}}"""


class VisionReasoner:
    """Optional Qwen2.5-VL step for flowcharts/charts whose logic OCR cannot express."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.processor = None

    @property
    def enabled(self) -> bool:
        return self.config.vision.enabled

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        except ImportError:  # Older transformers releases.
            from transformers import AutoModelForVision2Seq as ModelClass

        quantization_config = None
        if self.config.vision.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        self.processor = AutoProcessor.from_pretrained(
            self.config.vision.model,
            revision=self.config.vision.revision,
            trust_remote_code=True,
        )
        device_map = {"": self.config.vision.device} if self.config.vision.device else "auto"
        self.model = ModelClass.from_pretrained(
            self.config.vision.model,
            revision=self.config.vision.revision,
            trust_remote_code=True,
            device_map=device_map,
            torch_dtype="auto",
            quantization_config=quantization_config,
        )

    def describe(
        self,
        image_path: str | Path,
        image_type: str,
        ocr_text: str = "",
        caption: str = "",
        section: str = "",
        previous_context: str = "",
    ) -> dict[str, Any]:
        """Return ``{"status": "success"|"needs_review"|"skipped", "output": dict|None}``.

        ``previous_context`` is the end of the single preceding page (never more), used
        only to understand content that continues across the page break.

        Invalid JSON is retried a bounded number of times and then marked
        ``needs_review``; nothing is invented as a replacement.
        """
        if not self.enabled or image_type not in self.config.vision.image_types:
            return {"status": "skipped", "output": None}
        template = CHART_PROMPT if image_type == "chart" else FLOWCHART_PROMPT
        prompt = template.format(
            caption=caption or "(không có)",
            section=section or "(không rõ)",
            ocr_text=(ocr_text or "(trống)")[:3000],
            previous=(
                "Ngữ cảnh cuối trang liền trước (chỉ để hiểu phần nối tiếp; không chép thành nhãn của ảnh):\n"
                f"{previous_context.strip()[: self.config.vision.previous_page_chars]}\n"
                if previous_context.strip()
                else ""
            ),
        )
        errors = []
        for attempt in range(self.config.vision.max_retries + 1):
            try:
                raw = self._generate(image_path, prompt)
            except Exception as exc:  # GPU/model errors must not fail the whole document.
                LOGGER.exception("VLM failed for %s", image_path)
                return {"status": "failed", "output": None, "error": str(exc)}
            parsed = parse_json_object(raw)
            if parsed is not None and validate_vlm_output(parsed, image_type):
                return {"status": "success", "output": parsed, "attempts": attempt + 1}
            errors.append(raw[:500])
        return {"status": "needs_review", "output": None, "raw_attempts": errors}

    def unload(self) -> None:
        self.model = None
        self.processor = None
        release_gpu_memory()

    def _generate(self, image_path: str | Path, prompt: str) -> str:
        import torch
        from PIL import Image

        self.load()
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.config.vision.max_new_tokens, do_sample=False
            )
        output = generated[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(output, skip_special_tokens=True)[0].strip()


def parse_json_object(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def validate_vlm_output(output: dict[str, Any], image_type: str) -> bool:
    if image_type == "flowchart":
        return isinstance(output.get("nodes"), list) and isinstance(output.get("edges"), list)
    return isinstance(output.get("observations", []), list) and isinstance(output.get("summary", ""), str)


def render_vlm_output(output: dict[str, Any]) -> str:
    """Text for retrieval; observations and model commentary are labelled separately."""
    lines = []
    if output.get("title"):
        lines.append(f"Tiêu đề: {output['title']}")
    if output.get("image_type") == "flowchart" or "nodes" in output:
        labels = {node.get("id"): node.get("label", "") for node in output.get("nodes", []) if isinstance(node, dict)}
        if labels:
            lines.append("Các bước: " + "; ".join(f"{key}: {value}" for key, value in labels.items()))
        edges = []
        for edge in output.get("edges", []):
            if not isinstance(edge, dict):
                continue
            arrow = f"{labels.get(edge.get('from'), edge.get('from'))} → {labels.get(edge.get('to'), edge.get('to'))}"
            if edge.get("condition"):
                arrow += f" (nếu {edge['condition']})"
            edges.append(arrow)
        if edges:
            lines.append("Luồng xử lý:\n" + "\n".join(f"- {edge}" for edge in edges))
    else:
        if output.get("chart_type"):
            lines.append(f"Loại biểu đồ: {output['chart_type']}")
        for key, label in (("axes", "Trục"), ("legend", "Chú giải")):
            if output.get(key):
                lines.append(f"{label}: {', '.join(map(str, output[key]))}")
        if output.get("observations"):
            lines.append("Quan sát (từ ảnh/OCR):\n" + "\n".join(f"- {item}" for item in output["observations"]))
    if output.get("summary"):
        lines.append(f"Nhận xét do VLM suy luận (không dùng làm số liệu): {output['summary']}")
    if output.get("uncertainties"):
        lines.append("Điểm chưa chắc chắn: " + "; ".join(map(str, output["uncertainties"])))
    return "\n".join(lines)


def release_gpu_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
