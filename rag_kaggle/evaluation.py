from __future__ import annotations

import datetime as dt
import json
import statistics
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .guardrails import is_refusal
from .models import SearchHit
from .utils import extract_numbers, normalize_for_match, parse_cell_range, ranges_overlap

if TYPE_CHECKING:
    from .pipeline import RAGPipeline


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    items = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if "question" not in item:
                raise ValueError(f"Line {line_number} has no 'question'")
            items.append(item)
    return items


def expects_no_answer(item: dict[str, Any]) -> bool:
    return bool(item.get("no_answer")) or item.get("expected_answer") in (None, "") or item.get("category") == "no_answer"


def hit_matches(hit: SearchHit, item: dict[str, Any]) -> bool:
    """Whether a retrieved chunk is evidence for the dataset item."""
    chunk = hit.chunk
    expected_blocks = set(item.get("expected_block_ids") or [])
    if expected_blocks:
        return bool(expected_blocks & set(chunk.block_ids))
    expected_document = item.get("expected_document")
    if expected_document and normalize_for_match(expected_document) != normalize_for_match(chunk.source_file):
        return False
    location = item.get("expected_location") or {}
    if "page" in location:
        start = chunk.page_start
        end = chunk.page_end if chunk.page_end is not None else start
        if start is None or not (start <= int(location["page"]) <= end):
            return False
    if "sheet_name" in location and normalize_for_match(location["sheet_name"]) != normalize_for_match(chunk.sheet_name or ""):
        return False
    if "cell_range" in location:
        expected = parse_cell_range(location["cell_range"])
        actual = parse_cell_range(chunk.cell_range or "")
        if not expected or not actual or not ranges_overlap(expected[1:], actual[1:]):
            return False
    if "section" in location:
        if normalize_for_match(location["section"]) not in normalize_for_match(" > ".join(chunk.section_path)):
            return False
    return bool(expected_document or location)


def retrieval_metrics(results: list[tuple[dict, list[SearchHit]]], ks: tuple[int, ...]) -> dict[str, Any]:
    evaluated = [(item, hits) for item, hits in results if not expects_no_answer(item)]
    if not evaluated:
        return {"evaluated": 0}
    recall = {k: 0.0 for k in ks}
    reciprocal_ranks = []
    by_type: dict[str, list[int]] = {}
    coverage = []
    for item, hits in evaluated:
        matches = [hit_matches(hit, item) for hit in hits]
        first = next((index for index, matched in enumerate(matches) if matched), None)
        for k in ks:
            recall[k] += 1.0 if first is not None and first < k else 0.0
        reciprocal_ranks.append(1.0 / (first + 1) if first is not None else 0.0)
        content_type = item.get("content_type") or "unknown"
        by_type.setdefault(content_type, []).append(1 if first is not None and first < max(ks) else 0)
        expected_blocks = set(item.get("expected_block_ids") or [])
        if expected_blocks:
            found = set().union(*(set(hit.chunk.block_ids) for hit in hits[: max(ks)])) if hits else set()
            coverage.append(len(expected_blocks & found) / len(expected_blocks))
    count = len(evaluated)
    return {
        "evaluated": count,
        **{f"recall@{k}": round(recall[k] / count, 4) for k in ks},
        "mrr": round(sum(reciprocal_ranks) / count, 4),
        "hit_rate_by_content_type": {key: round(sum(values) / len(values), 4) for key, values in by_type.items()},
        "block_coverage": round(sum(coverage) / len(coverage), 4) if coverage else None,
    }


def answer_correct(answer: str, expected: str) -> bool:
    if not expected:
        return False
    normalized_answer = normalize_for_match(answer)
    if normalize_for_match(expected) in normalized_answer:
        return True
    expected_numbers = extract_numbers(expected)
    if expected_numbers:
        answer_numbers = {round(value, 6) for value in extract_numbers(answer)}
        return all(round(value, 6) in answer_numbers for value in expected_numbers)
    return False


def citation_correct(citations: list[str], item: dict[str, Any]) -> bool | None:
    expected_document = item.get("expected_document")
    if not expected_document:
        return None
    location = item.get("expected_location") or {}
    for citation in citations:
        text = normalize_for_match(citation)
        if normalize_for_match(expected_document) not in text:
            continue
        if "page" in location and f"trang {location['page']}" not in text and f"-{location['page']}" not in text:
            continue
        if "sheet_name" in location and normalize_for_match(location["sheet_name"]) not in text:
            continue
        return True
    return False


def answer_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if not row["expects_no_answer"]]
    unanswerable = [row for row in rows if row["expects_no_answer"]]
    numeric = [row for row in answerable if extract_numbers(row["expected_answer"] or "")]
    cited = [row for row in answerable if row["citation_correct"] is not None]

    def rate(values):
        return round(sum(values) / len(values), 4) if values else None

    return {
        "answered": len(rows),
        "correctness": rate([row["correct"] for row in answerable]),
        "numeric_exact_match": rate([row["correct"] for row in numeric]),
        "faithfulness_proxy": rate([not row["unsupported_numbers"] and row["citation_valid"] for row in answerable]),
        "citation_accuracy": rate([row["citation_correct"] for row in cited]),
        "refusal_accuracy": rate([row["refused"] for row in unanswerable]),
        "false_refusal_rate": rate([row["refused"] for row in answerable]),
    }


def run_evaluation(
    pipeline: "RAGPipeline",
    dataset_path: str | Path,
    run_answers: bool = True,
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, Any]:
    items = load_dataset(dataset_path)
    retrieval_results = []
    answer_rows = []
    latencies: dict[str, list[float]] = {}
    for item in items:
        started = time.perf_counter()
        hits = pipeline.retriever.retrieve(item["question"], top_k=max(ks))
        latencies.setdefault("retrieve", []).append((time.perf_counter() - started) * 1000)
        retrieval_results.append((item, hits))
        if not run_answers:
            continue
        result = pipeline.ask(item["question"])
        for stage, value in (result.get("latency_ms") or {}).items():
            latencies.setdefault(f"ask.{stage}", []).append(value)
        answer = result.get("answer", "")
        answer_rows.append(
            {
                "question": item["question"],
                "category": item.get("category"),
                "expected_answer": item.get("expected_answer"),
                "answer": answer,
                "citations": result.get("citations", []),
                "expects_no_answer": expects_no_answer(item),
                "refused": bool(result.get("refused")) or is_refusal(answer),
                "correct": answer_correct(answer, item.get("expected_answer") or ""),
                "citation_valid": bool(result.get("citation_valid", True)),
                "citation_correct": citation_correct(result.get("citations", []), item),
                "unsupported_numbers": result.get("unsupported_numbers", []),
                "trace_id": result.get("trace_id"),
            }
        )

    report = {
        "dataset": str(dataset_path),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "questions": len(items),
        "retrieval": retrieval_metrics(retrieval_results, ks),
        "answer": answer_metrics(answer_rows) if run_answers else None,
        "by_category": _by_category(answer_rows) if run_answers else None,
        "operational": {
            "latency_ms_p50": {stage: round(statistics.median(values), 1) for stage, values in latencies.items()},
            "latency_ms_max": {stage: round(max(values), 1) for stage, values in latencies.items()},
            "ingestion": operational_metrics(pipeline),
            **gpu_memory(),
        },
        "rows": answer_rows,
    }
    output_dir = pipeline.config.work_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"report_{dt.datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(path)
    return report


def _by_category(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row.get("category") or "uncategorized", []).append(row)
    return {
        category: {
            "count": len(items),
            "correct": round(sum(1 for row in items if (row["refused"] if row["expects_no_answer"] else row["correct"])) / len(items), 4),
        }
        for category, items in groups.items()
    }


def operational_metrics(pipeline: "RAGPipeline") -> dict[str, Any]:
    statuses = pipeline.metadata.statuses(limit=5000)
    parse = [status for status in statuses if status["stage"] == "parse" and status["status"] != "skipped"]
    ocr = [status for status in statuses if status["stage"] in ("ocr", "vlm")]
    index = [status for status in statuses if status["stage"] == "index" and status["status"] == "success"]
    throughput = None
    if index:
        durations = [status["duration_ms"] for status in index if status["duration_ms"]]
        chunks = pipeline.metadata.stats()["chunks"]
        throughput = round(chunks / (sum(durations) / 1000), 2) if durations and sum(durations) else None
    return {
        "parse_success_rate": round(sum(status["status"] in ("success", "partial") for status in parse) / len(parse), 4)
        if parse
        else None,
        "ocr_vlm_issue_count": len(ocr),
        "indexing_chunks_per_second": throughput,
    }


def gpu_memory() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            return {"peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
    except ImportError:
        pass
    return {}
