from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
WORKER_PREFIX = "@@OCR@@ "


def pipeline_version_for(model_name: str) -> str | None:
    """Map ``PaddleOCR-VL-1.6`` -> ``v1.6`` (PaddleOCRVL ``pipeline_version``)."""
    match = re.search(r"PaddleOCR-VL(?:-(\d+(?:\.\d+)?))?", model_name or "", re.IGNORECASE)
    if not match:
        return None
    return f"v{match.group(1)}" if match.group(1) else "v1"


class PaddleOCRVLAdapter:
    """Compatibility layer around PaddleOCR-VL releases.

    Two modes:

    * in-process (``python_executable=None``): imports ``paddleocr`` directly;
    * worker (``python_executable`` = Python of a separate venv): PaddlePaddle runs
      in its own process, so its CUDA/cuDNN wheels never conflict with PyTorch and
      ``unload()`` releases all of its GPU memory by terminating the process.

    The raw model result is preserved for audit and re-parsing.
    """

    def __init__(
        self,
        model_name: str,
        model_dir: str | None = None,
        python_executable: str | None = None,
        log_path: str | Path | None = None,
    ):
        self.model_name = model_name
        self.model_dir = model_dir
        self.python_executable = python_executable
        self.log_path = Path(log_path) if log_path else None
        self.pipeline = None
        self.load_error: str | None = None
        self._worker: subprocess.Popen | None = None
        self._worker_log = None

    # ------------------------------------------------------------------ load

    def load(self) -> None:
        if self.load_error:
            # Do not retry a failed model load for every page/image of a corpus.
            raise RuntimeError(self.load_error)
        try:
            if self.python_executable:
                self._start_worker()
            else:
                self._load_in_process()
        except Exception as exc:
            self.load_error = f"PaddleOCR-VL unavailable: {exc}"
            raise RuntimeError(self.load_error) from exc

    def _load_in_process(self) -> None:
        if self.pipeline is not None:
            return
        try:
            from paddleocr import PaddleOCRVL
        except ImportError as exc:
            raise RuntimeError(
                "paddleocr is not installed in this Python environment. Run the PaddleOCR "
                "install cell of the Kaggle notebook or set parsing.ocr_python."
            ) from exc

        version = pipeline_version_for(self.model_name)
        attempts: list[dict[str, Any]] = []
        preferred: dict[str, Any] = {}
        if version:
            preferred["pipeline_version"] = version
        if self.model_dir:
            preferred["vl_rec_model_dir"] = self.model_dir
        attempts.append(preferred)
        if self.model_dir:
            attempts.append({"vl_rec_model_dir": self.model_dir})
        attempts.append({})

        errors = []
        for kwargs in attempts:
            try:
                self.pipeline = PaddleOCRVL(**kwargs)
                if not kwargs or "pipeline_version" not in kwargs:
                    LOGGER.warning(
                        "PaddleOCRVL rejected pipeline_version=%s; using the package default model. "
                        "Pin paddleocr to guarantee %s.",
                        version,
                        self.model_name,
                    )
                return
            except Exception as exc:  # Constructor signatures differ between releases.
                errors.append(f"{kwargs}: {type(exc).__name__}: {exc}")
        raise RuntimeError("Unable to initialize PaddleOCR-VL:\n" + "\n".join(errors))

    # --------------------------------------------------------------- predict

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        self.load()
        if self.python_executable:
            return self._predict_with_worker(image_path)
        results = list(self.pipeline.predict(str(image_path)))
        return self.process_results(results)

    def process_results(self, results: list[Any]) -> dict[str, Any]:
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
        if self._worker is not None:
            self._stop_worker()
            return
        if self.pipeline is None:
            return
        self.pipeline = None
        try:
            import gc
            import paddle

            gc.collect()
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
        except Exception:
            pass

    # ---------------------------------------------------------------- worker

    def _start_worker(self) -> None:
        if self._worker is not None and self._worker.poll() is None:
            return
        if not Path(self.python_executable).exists():
            raise RuntimeError(f"OCR Python not found: {self.python_executable}")
        command = [self.python_executable, "-u", str(Path(__file__).resolve()), "--worker", "--model-name", self.model_name]
        if self.model_dir:
            command += ["--model-dir", self.model_dir]
        env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "PYTHONHOME")}
        env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._worker_log = self.log_path.open("a", encoding="utf-8")
        self._worker = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._worker_log or subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=env,
        )
        ready = self._read_worker_message()
        if not ready.get("ready"):
            error = ready.get("error", "unknown error")
            self._stop_worker()
            raise RuntimeError(f"OCR worker failed to load model: {error}")
        LOGGER.info("PaddleOCR-VL worker ready: %s", ready.get("info"))

    def _predict_with_worker(self, image_path: str | Path) -> dict[str, Any]:
        self._worker.stdin.write(json.dumps({"image": str(image_path)}) + "\n")
        self._worker.stdin.flush()
        message = self._read_worker_message()
        if "error" in message:
            raise RuntimeError(message["error"])
        return message["result"]

    def _read_worker_message(self) -> dict[str, Any]:
        while True:
            line = self._worker.stdout.readline()
            if not line:
                code = self._worker.poll()
                raise RuntimeError(
                    f"OCR worker exited (code={code}). See {self.log_path or 'worker stderr'} for details."
                )
            if line.startswith(WORKER_PREFIX):
                return json.loads(line[len(WORKER_PREFIX) :])

    def _stop_worker(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            try:
                worker.stdin.close()
                worker.wait(timeout=30)
            except Exception:
                worker.kill()
        if self._worker_log is not None:
            self._worker_log.close()
            self._worker_log = None

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


def _worker_main(argv: list[str]) -> int:
    """Run inside the PaddleOCR venv: JSON lines in on stdin, prefixed JSON out on stdout."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--model-name", default="PaddleOCR-VL-1.6")
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args(argv)

    # Keep the protocol channel clean: anything libraries print goes to stderr.
    protocol = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(payload: dict[str, Any]) -> None:
        protocol.write(WORKER_PREFIX + json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        protocol.flush()

    adapter = PaddleOCRVLAdapter(args.model_name, args.model_dir)
    try:
        adapter._load_in_process()
        import paddle

        info = {"paddle": paddle.__version__, "cuda": paddle.device.is_compiled_with_cuda()}
        try:
            import paddleocr

            info["paddleocr"] = getattr(paddleocr, "__version__", "unknown")
        except Exception:
            pass
        send({"ready": True, "info": info})
    except Exception as exc:
        send({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            results = list(adapter.pipeline.predict(request["image"]))
            send({"result": adapter.process_results(results)})
        except Exception as exc:
            send({"error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1:]))
