import importlib.util
import tempfile
import unittest
from pathlib import Path

from rag_kaggle.config import CorrectionConfig, PipelineConfig
from rag_kaggle.correction import (
    SYSTEM_PROMPT,
    DocumentMemory,
    OCRCorrector,
    accept_correction,
    fold_accents,
    parse_response,
)
from rag_kaggle.hardware import configure_ingestion_devices
from rag_kaggle.knowledge import build_document_profile, format_profile, match_terms, profile_terms
from rag_kaggle.models import Block, ParsedDocument
from rag_kaggle.parsers import DocumentParser


def reply(text, entities=""):
    return f"<corrected>\n{text}\n</corrected>\n<entities>\n{entities}\n</entities>"


class FakeLLM:
    """Stands in for the GPU model; ``script`` maps the prompt to a raw reply."""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.unloaded = False

    def _chat(self, system_prompt, user_prompt, max_new_tokens=None, repetition_penalty=None):
        self.calls.append(
            {"system": system_prompt, "user": user_prompt, "max_new_tokens": max_new_tokens, "penalty": repetition_penalty}
        )
        result = self.script(user_prompt) if callable(self.script) else self.script
        if isinstance(result, Exception):
            raise result
        return result

    def unload(self):
        self.unloaded = True


def make_corrector(script, **overrides):
    config = PipelineConfig()
    config.correction.enabled = True
    for key, value in overrides.items():
        setattr(config.correction, key, value)
    llm = FakeLLM(script)
    corrector = OCRCorrector(config, llm=llm)
    corrector.begin_document("doc.pdf")
    return corrector, llm


def target_of(prompt):
    return prompt.split("ĐOẠN CẦN SỬA", 1)[1].split(":\n", 1)[1]


class CorrectionTests(unittest.TestCase):
    def test_fixes_text_keeps_original_and_learns_entities(self):
        corrector, _ = make_corrector(
            reply("Ông Nguyễn Văn Hùng ký quyết định tại Hà Nội.", "person: Nguyễn Văn Hùng\nplace: Hà Nội\nperson: Người Bịa Đặt")
        )
        result = corrector.correct("Ông Nguyen Văn Hung ky quyêt dinh tại Hà Nôi.", page=1)
        self.assertTrue(result.changed)
        self.assertEqual("corrected", result.status)
        self.assertEqual("Ông Nguyen Văn Hung ky quyêt dinh tại Hà Nôi.", result.original)
        memory = corrector.finish_document()["memory"]
        self.assertEqual(["Nguyễn Văn Hùng"], [group["value"] for group in memory["person"]])  # invented name ignored
        self.assertEqual(["Hà Nội"], [group["value"] for group in memory["place"]])

    def test_each_call_holds_only_instruction_glossary_previous_and_target(self):
        paragraphs = [f"Đoạn số {name} nói về quy trình thẩm định hồ sơ." for name in ("một", "hai", "ba")]
        fixed = [text.replace("Đoạn", "Đoạn") + " (đã sửa)" for text in paragraphs]
        queue = iter(fixed)
        corrector, llm = make_corrector(lambda prompt: reply(next(queue), "keyword: thẩm định hồ sơ"), max_length_change=1)
        for page, text in enumerate(paragraphs, start=1):
            corrector.correct(text, page=page)
        third = llm.calls[2]["user"]
        self.assertIn("ĐOẠN TRƯỚC", third)
        self.assertIn(fixed[1], third)  # the previous paragraph, in its corrected form
        self.assertNotIn(paragraphs[0], third)  # nothing older than one block
        self.assertNotIn(fixed[0], third)
        self.assertIn("- keyword: thẩm định hồ sơ", third)  # glossary
        self.assertTrue(all(call["system"] == SYSTEM_PROMPT for call in llm.calls))
        self.assertTrue(all(call["penalty"] == 1.0 for call in llm.calls))
        self.assertLess(len(third), len(SYSTEM_PROMPT) + 1500)  # bounded regardless of document length

    def test_context_never_reaches_further_back_than_one_page(self):
        corrector, llm = make_corrector(lambda prompt: reply(target_of(prompt).strip()), context_blocks=5)
        for page, text in ((1, "Nội dung trang một khá dài."), (2, "Nội dung trang hai khá dài."), (4, "Nội dung trang bốn khá dài.")):
            corrector.correct(text, page=page)
        self.assertIn("trang một", llm.calls[1]["user"])  # page 1 -> page 2: allowed
        self.assertNotIn("trang một", llm.calls[2]["user"])  # page 4 may not see page 1 ...
        self.assertNotIn("trang hai", llm.calls[2]["user"])  # ... nor page 2

    def test_new_document_forgets_everything(self):
        corrector, llm = make_corrector(lambda prompt: reply(target_of(prompt).strip(), "person: Trần Thị Bình"))
        corrector.correct("Bà Trần Thị Bình phụ trách hồ sơ này.", page=1)
        self.assertTrue(corrector.memory)
        corrector.begin_document("other.pdf")
        corrector.correct("Một đoạn hoàn toàn khác của tài liệu khác.", page=1)
        self.assertNotIn("Trần Thị Bình", llm.calls[-1]["user"])
        self.assertNotIn("ĐOẠN TRƯỚC", llm.calls[-1]["user"])

    def test_unsafe_rewrites_are_rejected_and_original_kept(self):
        original = "Phí thường niên là 499.000 đồng, hiệu lực từ ngày 01/01/2025."
        for bad, reason in (
            (original.replace("499.000", "500.000"), "numbers_changed"),
            ("Tóm lại, phí thường niên rất thấp.", "numbers_changed"),
            (original + " " + "Thêm một câu giải thích dài dòng không có trong văn bản gốc. " * 3, "length_changed"),
            ("", "empty"),
        ):
            corrector, _ = make_corrector(reply(bad))
            result = corrector.correct(original, page=1)
            self.assertEqual(original, result.text, reason)
            self.assertFalse(result.changed)
            self.assertEqual("rejected", result.status)
            self.assertEqual(1, corrector.finish_document()["reject_reasons"][reason])

    def test_accept_correction_similarity_and_identity(self):
        config = CorrectionConfig()
        self.assertEqual((True, "unchanged"), accept_correction("abc def", "abc def", config))
        self.assertEqual((False, "too_different"), accept_correction("Quy trình phê duyệt hồ sơ", "Đây là nội dung khác hẳn", config))
        self.assertEqual((True, "corrected"), accept_correction("Quy trinh phe duyet ho so", "Quy trình phê duyệt hồ sơ", config))

    def test_repeated_failures_disable_correction_without_raising(self):
        corrector, llm = make_corrector(RuntimeError("CUDA out of memory"), max_consecutive_failures=2)
        texts = [f"Đoạn văn bản số {index} cần được sửa lỗi." for index in range(4)]
        results = [corrector.correct(text, page=1) for text in texts]
        self.assertEqual(texts, [result.text for result in results])
        self.assertEqual(2, len(llm.calls))  # stopped calling the model after the 2nd consecutive failure
        self.assertFalse(corrector.enabled)
        self.assertIn("out of memory", corrector.finish_document()["disabled_reason"])

    def test_long_text_is_corrected_in_bounded_segments(self):
        corrector, llm = make_corrector(lambda prompt: reply(target_of(prompt).strip()), segment_chars=300)
        text = "\n\n".join(f"Đoạn thứ {index} nêu rõ điều kiện áp dụng của sản phẩm tín dụng." * 2 for index in range(8))
        result = corrector.correct(text, page=1)
        self.assertGreater(len(llm.calls), 1)
        self.assertTrue(all(len(target_of(call["user"])) <= 320 for call in llm.calls))
        self.assertEqual("unchanged", result.status)

    def test_disabled_or_short_text_never_calls_the_model(self):
        corrector, llm = make_corrector(reply("x"))
        self.assertEqual("skipped", corrector.correct("12", page=1).status)
        self.assertEqual("skipped", corrector.correct("2025 / 01 / 02 / 03", page=1).status)
        corrector.config.enabled = False
        self.assertEqual("Một đoạn đủ dài để sửa.", corrector.correct("Một đoạn đủ dài để sửa.", page=1).text)
        self.assertEqual([], llm.calls)

    def test_parse_response_is_tolerant(self):
        self.assertEqual(("Xin chào", [("place", "Huế")]), parse_response(reply("Xin chào", "- place: Huế\nlinh tinh: bỏ qua")))
        self.assertEqual(("Chỉ có văn bản", []), parse_response("<corrected>Chỉ có văn bản"))
        self.assertEqual(("Không thẻ", []), parse_response("```\nKhông thẻ\n```"))
        self.assertEqual(("Văn bản", []), parse_response("<corrected>Văn bản</corrected>"))

    def test_memory_is_bounded_and_reports_diacritic_variants(self):
        config = CorrectionConfig(memory_max_chars=120, memory_max_items_per_kind=3)
        memory = DocumentMemory(config)
        for _ in range(3):
            memory.add("person", "Nguyễn Văn Hùng")
        memory.add("person", "Nguyễn Văn Hưng")
        for index in range(10):
            memory.add("place", f"Địa danh số {index}")
        self.assertEqual(3, len(memory.groups("place")))
        group = memory.groups("person")[0]
        self.assertEqual("Nguyễn Văn Hùng", group["value"])
        self.assertEqual(["Nguyễn Văn Hưng"], group["variants"])  # reported, not silently merged
        self.assertLessEqual(len(memory.render_for_prompt()), 120)
        self.assertEqual("nguyen van hung", fold_accents("Nguyễn Văn Hùng"))
        self.assertFalse(memory.add("person", "12345"))
        self.assertFalse(memory.add("unknown", "Tên"))

    def test_device_planner_puts_correction_on_the_idle_gpu(self):
        config = PipelineConfig()
        layout = configure_ingestion_devices(config, gpu_count=2)
        self.assertEqual("cuda:0", config.correction.device)
        self.assertEqual("1", config.parsing.ocr_cuda_visible_devices)  # OCR stays on the other GPU
        self.assertIn("OCR correction LLM (during parsing)", layout["gpu_0"])
        cpu = PipelineConfig()
        cpu.correction.enabled = True
        configure_ingestion_devices(cpu, gpu_count=0)
        self.assertFalse(cpu.correction.enabled)


def has(*modules):
    return all(importlib.util.find_spec(module) is not None for module in modules)


class FakeOCR:
    def predict(self, image_path):
        blocks = [
            {"label": "doc_title", "content": "QUYET DINH SO 12/QD-UBND", "bbox": None, "confidence": 0.9},
            {"label": "text", "content": "Ong Nguyen Van A ky quyet dinh tai Ha Noi.", "bbox": None, "confidence": 0.9},
            {"label": "table", "content": "| Ma | Phi |\n| --- | --- |\n| A1 | 499 |", "bbox": None, "confidence": 0.9},
            {"label": "formula", "content": "x = y + 1", "bbox": None, "confidence": 0.9},
        ]
        return {"text": "\n\n".join(block["content"] for block in blocks), "blocks": blocks, "raw": [], "model": "fake"}


@unittest.skipUnless(has("fitz", "PIL"), "PyMuPDF/Pillow missing")
class ParserIntegrationTests(unittest.TestCase):
    def test_scanned_page_text_is_corrected_but_tables_and_formulas_are_not(self):
        import fitz

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "scan.pdf"
            pdf = fitz.open()
            pdf.new_page(width=400, height=400)  # No text layer -> OCR path.
            pdf.save(pdf_path)
            pdf.close()
            config = PipelineConfig(work_dir=Path(tmp) / "work")
            config.create_directories()
            config.correction.enabled = True
            fixes = {
                "QUYET DINH SO 12/QD-UBND": reply("QUYẾT ĐỊNH SỐ 12/QĐ-UBND"),
                "Ong Nguyen Van A ky quyet dinh tai Ha Noi.": reply(
                    "Ông Nguyễn Văn A ký quyết định tại Hà Nội.", "signer: Nguyễn Văn A\nplace: Hà Nội"
                ),
            }
            llm = FakeLLM(lambda prompt: fixes[target_of(prompt).strip()])
            corrector = OCRCorrector(config, llm=llm)
            parser = DocumentParser(config, ocr=FakeOCR(), corrector=corrector)

            document = parser.parse(pdf_path)

            by_type = {}
            for block in document.blocks:
                by_type.setdefault(block.block_type, []).append(block)
            heading, paragraph, formula = by_type["heading"][0], by_type["text"][0], by_type["text"][1]
            self.assertEqual("QUYẾT ĐỊNH SỐ 12/QĐ-UBND", heading.content)
            self.assertEqual("QUYET DINH SO 12/QD-UBND", heading.metadata["ocr_original_text"])
            self.assertEqual("Ông Nguyễn Văn A ký quyết định tại Hà Nội.", paragraph.content)
            self.assertEqual("corrected", paragraph.metadata["ocr_correction"])
            self.assertEqual("x = y + 1", formula.content)
            self.assertNotIn("ocr_correction", formula.metadata)
            self.assertIn("| A1 | 499 |", by_type["table"][0].content)
            self.assertEqual(2, len(llm.calls))  # table and formula never reached the model
            self.assertIn("QUYẾT ĐỊNH SỐ 12/QĐ-UBND", llm.calls[1]["user"])  # previous paragraph, corrected
            info = document.metadata["ocr_correction"]
            self.assertEqual(2, info["corrected"])
            self.assertEqual("Nguyễn Văn A", info["memory"]["signer"][0]["value"])

            profile = build_document_profile(document)
            self.assertEqual(["12/QĐ-UBND"], profile["doc_numbers"])
            self.assertEqual(["Nguyễn Văn A"], profile["entities"]["signer"])
            self.assertEqual(["Hà Nội"], profile["entities"]["place"])


class KnowledgeTests(unittest.TestCase):
    def test_profile_from_rules_and_glossary(self):
        document = ParsedDocument("d", "qd.pdf", "pdf", "h")
        document.blocks = [
            Block("b1", "d", "heading", "QUYẾT ĐỊNH VỀ VIỆC BAN HÀNH QUY CHẾ", "qd.pdf"),
            Block("b2", "d", "text", "Số: 45/2025/QĐ-UBND\nHà Nội, ngày 5 tháng 3 năm 2025", "qd.pdf"),
            Block("b3", "d", "text", "Người ký: Trần Thị Bình", "qd.pdf"),
            Block("b4", "d", "heading", "I.", "qd.pdf"),
            Block("b5", "d", "heading", "Phạm vi điều chỉnh", "qd.pdf"),
        ]
        document.metadata["ocr_correction"] = {"memory": {"place": [{"value": "Hà Nội", "count": 2, "variants": []}]}}
        profile = build_document_profile(document)
        self.assertEqual("QUYẾT ĐỊNH VỀ VIỆC BAN HÀNH QUY CHẾ", profile["title"])
        self.assertEqual(["45/2025/QĐ-UBND"], profile["doc_numbers"])
        self.assertEqual(["05/03/2025"], profile["dates"])
        self.assertEqual(["Trần Thị Bình"], profile["entities"]["signer"])
        self.assertEqual(["Hà Nội"], profile["entities"]["place"])
        self.assertEqual(["QUYẾT ĐỊNH VỀ VIỆC BAN HÀNH QUY CHẾ", "Phạm vi điều chỉnh"], profile["keywords"])
        line = format_profile(profile)
        self.assertIn("số hiệu: 45/2025/QĐ-UBND", line)
        self.assertIn("người ký: Trần Thị Bình", line)

    def test_term_matching_respects_word_boundaries(self):
        terms = profile_terms({"entities": {"person": ["An"], "place": ["Hà Nội"]}})
        self.assertEqual(["Hà Nội"], match_terms(terms["entities"], "Cơ quan đặt tại hà nội (HÀ NỘI).", 5))
        self.assertEqual([], match_terms(terms["entities"], "Bản án tuyên phạt", 5))  # "an" inside other words
        self.assertEqual(["An"], match_terms(terms["entities"], "Ông An đã ký", 5))


if __name__ == "__main__":
    unittest.main()
