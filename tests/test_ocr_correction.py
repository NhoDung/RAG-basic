import threading
import unittest

from rag_kaggle.config import PipelineConfig
from rag_kaggle.entities import EntityLedger, extract_entity_candidates, strip_diacritics
from rag_kaggle.models import Block, ParsedDocument
from rag_kaggle.ocr_correction import (
    OCRCorrectionCoordinator,
    build_correction_prompt,
    split_segments,
    table_to_text,
    tail_context,
    text_to_rows,
)
from rag_kaggle.parsers import DocumentParser


class ScriptedCorrector:
    """Appends a marker; optionally blocks, fails or mangles tables for a given input."""

    def __init__(self, fail_on=None, gate=None, mangle_tables=False):
        self.calls = []
        self.fail_on = fail_on
        self.gate = gate
        self.mangle_tables = mangle_tables

    def correct(self, previous, current, glossary=(), table=False):
        self.calls.append({"previous": previous, "current": current, "glossary": tuple(glossary), "table": table})
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail_on and self.fail_on in current:
            raise RuntimeError("boom")
        if table and self.mangle_tables:
            return current.split("\n")[0], {"prompt_tokens": 1, "completion_tokens": 1}
        return current.replace("sai", "đúng"), {"prompt_tokens": 1, "completion_tokens": 1}

    def unload(self):
        pass


def make_config(**overrides):
    config = PipelineConfig()
    config.ocr_correction.enabled = True
    for key, value in overrides.items():
        setattr(config.ocr_correction, key, value)
    return config


def text_block(document_id, content, block_type="text", **kwargs):
    return Block(f"b-{content[:8]}", document_id, block_type, content, "scan.pdf", metadata={}, **kwargs)


class SegmentationTests(unittest.TestCase):
    def test_segments_concatenate_back_to_the_original(self):
        text = "Câu một. Câu hai dài hơn một chút! Câu ba?\nDòng mới không dấu chấm " + "x" * 50
        for limit in (20, 40, 80):
            segments = split_segments(text, limit)
            self.assertEqual(text, "".join(segments))
            self.assertTrue(all(len(segment) <= limit for segment in segments), segments)

    def test_table_segments_never_cut_a_row(self):
        text = "\n".join(f"a{i} | b{i} | 1.5 | x" for i in range(10))
        segments = split_segments(text, 40, by_lines=True)
        self.assertEqual(text, "".join(segments))
        for segment in segments:
            for line in segment.strip().split("\n"):
                self.assertEqual(3, line.count("|"))

    def test_tail_context_is_bounded(self):
        text = "Mot. Hai ba bon nam sau. Bay tam chin muoi."
        tail = tail_context(text, 20)
        self.assertLessEqual(len(tail), 20)
        self.assertTrue(text.endswith(tail))
        self.assertEqual("", tail_context(text, 0))

    def test_prompt_adds_glossary_and_table_hint_only_when_needed(self):
        _, plain = build_correction_prompt("prev", "cur")
        self.assertNotIn("DOCUMENT_TERMS", plain)
        _, rich = build_correction_prompt("prev", "cur", ["Hòa Bình"], table=True)
        self.assertIn("DOCUMENT_TERMS:\nHòa Bình", rich)
        self.assertIn("one row per line", rich)

    def test_table_text_roundtrip_preserves_special_cells(self):
        rows = [["a|b", "x\ny"], ["", "2"]]
        text = table_to_text(rows)
        self.assertEqual(rows, text_to_rows(text, [2, 2]))
        self.assertIsNone(text_to_rows(text, [2, 3]))
        self.assertIsNone(text_to_rows("only one line", [2, 2]))


class EntityTests(unittest.TestCase):
    def test_candidates_are_title_case_runs(self):
        found = extract_entity_candidates("Theo Nguyễn Văn An, xã Hòa Bình đã duyệt. UBND ký.")
        self.assertIn("Nguyễn Văn An", found)
        self.assertIn("Hòa Bình", found)
        self.assertNotIn("Theo Nguyễn Văn An", found)

    def test_ledger_picks_the_best_attested_spelling(self):
        ledger = EntityLedger(min_count=2)
        ledger.add("Ông Nguyễn Văn An ký. Nguyễn Văn An là giám đốc.")
        ledger.add("Nguyen Van An đã nhận.")
        self.assertEqual(["Nguyễn Văn An"], ledger.terms())
        self.assertEqual({"Nguyễn Văn An": ["Nguyen Van An"]}, ledger.conflicts())
        self.assertEqual("nguyen van an", strip_diacritics("Nguyễn Văn An").lower())


class CoordinatorTests(unittest.TestCase):
    def test_worker_error_unblocks_producer_and_worker_recovers(self):
        config = make_config(max_buffered_components=1, max_retries=0, fail_open=False)
        corrector = ScriptedCorrector(fail_on="XXX")
        coordinator = OCRCorrectionCoordinator(config, corrector=corrector)
        try:
            coordinator.begin_document("doc-a")
            outcome = []

            def produce():
                try:
                    for text in ("XXX one", "two", "three", "four"):
                        coordinator.submit(text_block("doc-a", text), "text")
                    coordinator.finish_document("doc-a")
                    outcome.append("finished")
                except RuntimeError as exc:
                    outcome.append(str(exc))

            thread = threading.Thread(target=produce, daemon=True)
            thread.start()
            thread.join(5)
            self.assertFalse(thread.is_alive(), "producer deadlocked after worker failure")
            self.assertEqual(["OCR correction worker failed"], outcome[:1])
            coordinator.abort_document("doc-a")

            block = text_block("doc-b", "dòng sai")
            coordinator.begin_document("doc-b")
            coordinator.submit(block, "text")
            coordinator.finish_document("doc-b")
            self.assertEqual("dòng đúng", block.content)
        finally:
            coordinator.close()

    def test_abort_mid_flight_does_not_corrupt_the_next_document(self):
        gate = threading.Event()
        config = make_config()
        corrector = ScriptedCorrector(gate=gate)
        coordinator = OCRCorrectionCoordinator(config, corrector=corrector)
        try:
            coordinator.begin_document("doc-a")
            stale = text_block("doc-a", "cũ sai")
            coordinator.submit(stale, "text")
            while not corrector.calls:  # Worker is now blocked inside the model call.
                threading.Event().wait(0.01)
            coordinator.abort_document("doc-a")
            fresh = text_block("doc-b", "mới sai")
            coordinator.begin_document("doc-b")
            coordinator.submit(fresh, "text")
            gate.set()
            stats = coordinator.finish_document("doc-b")
            self.assertEqual(1, stats["component_count"])
            self.assertEqual(1, stats["success_count"])
            self.assertEqual("mới đúng", fresh.content)
            self.assertEqual("", corrector.calls[-1]["previous"])
        finally:
            gate.set()
            coordinator.close()

    def test_long_component_is_corrected_in_bounded_segments(self):
        config = make_config(segment_chars=60, previous_context_chars=25)
        corrector = ScriptedCorrector()
        coordinator = OCRCorrectionCoordinator(config, corrector=corrector)
        try:
            text = " ".join(f"Câu số {i} sai chính tả." for i in range(12))
            block = text_block("doc-a", text)
            coordinator.begin_document("doc-a")
            coordinator.submit(block, "text")
            stats = coordinator.finish_document("doc-a")
            self.assertGreater(stats["segment_count"], 2)
            self.assertEqual(text.replace("sai", "đúng"), block.content)
            self.assertTrue(all(len(call["current"]) <= 60 for call in corrector.calls))
            self.assertTrue(all(len(call["previous"]) <= 25 for call in corrector.calls))
        finally:
            coordinator.close()

    def test_retry_drops_context_and_glossary(self):
        config = make_config(max_retries=1, glossary_enabled=True)
        corrector = ScriptedCorrector(fail_on="LỖI")
        coordinator = OCRCorrectionCoordinator(config, corrector=corrector)
        try:
            coordinator.begin_document("doc-a")
            coordinator.submit(text_block("doc-a", "Ông Nguyễn Văn An. Nguyễn Văn An ký."), "text")
            coordinator.submit(text_block("doc-a", "LỖI nặng"), "text")
            stats = coordinator.finish_document("doc-a")
            first, second = corrector.calls[1], corrector.calls[2]
            self.assertEqual(("Nguyễn Văn An",), first["glossary"])
            self.assertNotEqual("", first["previous"])
            self.assertEqual("", second["previous"])
            self.assertEqual((), second["glossary"])
            self.assertEqual(1, stats["fallback_count"])
            self.assertEqual(["Nguyễn Văn An"], stats["glossary_terms"])
        finally:
            coordinator.close()

    def test_table_cells_are_corrected_and_written_back_to_rows(self):
        config = make_config(correct_tables=True)
        coordinator = OCRCorrectionCoordinator(config, corrector=ScriptedCorrector())
        try:
            rows = [["Tên sai", "Số"], ["Hà Nội", "1.5"]]
            block = text_block("doc-a", "table", block_type="table")
            block.metadata["rows"] = [list(row) for row in rows]
            coordinator.begin_document("doc-a")
            coordinator.submit(block, "table")
            coordinator.finish_document("doc-a")
            self.assertEqual([["Tên đúng", "Số"], ["Hà Nội", "1.5"]], block.metadata["rows"])
            self.assertEqual(rows, block.metadata["rows_raw"])
            self.assertIn("Tên đúng", block.content)
        finally:
            coordinator.close()

    def test_table_structure_change_keeps_original_rows(self):
        config = make_config(correct_tables=True, max_retries=0)
        coordinator = OCRCorrectionCoordinator(config, corrector=ScriptedCorrector(mangle_tables=True))
        try:
            rows = [["A", "B"], ["1", "2"], ["3", "4"]]
            block = text_block("doc-a", "table", block_type="table")
            block.metadata["rows"] = [list(row) for row in rows]
            coordinator.begin_document("doc-a")
            coordinator.submit(block, "table")
            stats = coordinator.finish_document("doc-a")
            self.assertEqual(rows, block.metadata["rows"])
            self.assertNotIn("rows_raw", block.metadata)
            self.assertEqual("fallback", block.metadata["ocr_correction"]["status"])
            self.assertEqual(1, stats["fallback_count"])
        finally:
            coordinator.close()

    def test_blank_component_is_skipped_without_model_call(self):
        config = make_config()
        corrector = ScriptedCorrector()
        coordinator = OCRCorrectionCoordinator(config, corrector=corrector)
        try:
            coordinator.begin_document("doc-a")
            coordinator.submit(text_block("doc-a", "   "), "text")
            stats = coordinator.finish_document("doc-a")
            self.assertEqual(1, stats["skipped_count"])
            self.assertEqual([], corrector.calls)
        finally:
            coordinator.close()


class HeadingPropagationTests(unittest.TestCase):
    def test_corrected_heading_replaces_raw_text_in_section_paths(self):
        parser = DocumentParser(PipelineConfig())
        document = ParsedDocument("d", "scan.pdf", "pdf", "h")
        heading = Block(
            "h1", "d", "heading", "UY BAN NHAN DAN", "scan.pdf",
            section_path=["UY BAN NHAN DAN"],
            metadata={"ocr_text_raw": "UY BAN NHAN DAN", "ocr_correction": {"status": "success"}},
        )
        heading.content = "ỦY BAN NHÂN DÂN"
        child = Block("t1", "d", "text", "nội dung", "scan.pdf", section_path=["UY BAN NHAN DAN", "Mục 1"])
        document.blocks = [heading, child]
        parser._apply_corrected_headings(document)
        self.assertEqual(["ỦY BAN NHÂN DÂN"], heading.section_path)
        self.assertEqual(["ỦY BAN NHÂN DÂN", "Mục 1"], child.section_path)


if __name__ == "__main__":
    unittest.main()


class FakeVision:
    enabled = True

    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def describe(self, image_path, image_type, ocr_text="", caption="", section="", previous_context=""):
        self.calls.append({"ocr_text": ocr_text, "previous_context": previous_context, "path": image_path})
        output = self.outputs.pop(0)
        if output is None:
            return {"status": "needs_review", "output": None}
        return {"status": "success", "output": output}


class DeferredVisionTests(unittest.TestCase):
    def _document(self):
        from pathlib import Path

        from rag_kaggle.parsers import PendingVisual

        document = ParsedDocument("d", "scan.pdf", "pdf", "h")
        blocks = [
            Block("p1", "d", "text", "Trang một. " * 3 + "Phần cuối trang một.", "scan.pdf", page=1),
            Block("p0", "d", "text", "Trang không thuộc.", "scan.pdf", page=2),
            Block("img", "d", "image", "Bước 1 -> Bước 2 (đã sửa)", "scan.pdf", page=3, section_path=["A"]),
            Block("tail", "d", "text", "Sau ảnh.", "scan.pdf", page=3),
        ]
        # Page 2 text must not leak into the page-3 request; only page 2 is "previous".
        for index, block in enumerate(blocks):
            block.reading_order = index
        document.blocks = blocks
        pending = PendingVisual(blocks[2], Path("x.png"), "flowchart", "", ["A"], "PDF page 3", "ord-1", drop_if_empty=False)
        return document, pending

    def test_vlm_runs_after_correction_with_exactly_one_previous_page(self):
        document, pending = self._document()
        vision = FakeVision([{"image_type": "flowchart", "nodes": [{"id": "1", "label": "Bước 1"}], "edges": []}])
        parser = DocumentParser(PipelineConfig(), vision=vision)
        parser._deferred_vision = [pending]
        parser._run_deferred_vision(document)
        call = vision.calls[0]
        self.assertEqual("Bước 1 -> Bước 2 (đã sửa)", call["ocr_text"])  # corrected OCR text
        self.assertIn("Trang không thuộc", call["previous_context"])
        self.assertNotIn("Phần cuối trang một", call["previous_context"])  # page 1 is two pages back
        self.assertEqual(["p1", "p0", "img"], [b.block_id for b in document.blocks][:3])
        self.assertEqual("flowchart", document.blocks[3].block_type)
        self.assertEqual("tail", document.blocks[4].block_id)
        derived = document.blocks[3]
        self.assertEqual("img", derived.metadata["derived_from"])
        self.assertEqual([0, 1, 2, 3, 4], [b.reading_order for b in document.blocks])

    def test_empty_image_without_description_is_removed(self):
        document, pending = self._document()
        pending.drop_if_empty = True
        vision = FakeVision([None])
        parser = DocumentParser(PipelineConfig(), vision=vision)
        parser._deferred_vision = [pending]
        parser._run_deferred_vision(document)
        self.assertEqual("", vision.calls[0]["ocr_text"])
        self.assertNotIn("img", [b.block_id for b in document.blocks])
        self.assertEqual([0, 1, 2], [b.reading_order for b in document.blocks])

    def test_needs_review_marks_existing_image_block(self):
        document, pending = self._document()
        parser = DocumentParser(PipelineConfig(), vision=FakeVision([None]))
        parser._deferred_vision = [pending]
        parser._run_deferred_vision(document)
        self.assertEqual("needs_review", pending.block.metadata["vlm_status"])
