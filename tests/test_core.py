import unittest
from pathlib import Path

from rag_kaggle.chunking import ParentChildChunker
from rag_kaggle.config import PipelineConfig
from rag_kaggle.models import Block, ParsedDocument
from rag_kaggle.paddleocr_vl import PaddleOCRVLAdapter
from rag_kaggle.parsers import rows_to_markdown
from rag_kaggle.storage import MetadataStore, qdrant_point_id, tokenize_vi


class CorePipelineTests(unittest.TestCase):
    def test_parent_child_chunking(self):
        config = PipelineConfig()
        document = ParsedDocument("doc1", "sample.xlsx", "xlsx", "hash")
        rows = [["Loại thẻ", "Phí"], ["Visa Gold", "499000"]]
        document.blocks = [
            Block(
                "b1",
                "doc1",
                "text",
                "Quy trình phê duyệt hồ sơ tín dụng.",
                "sample.xlsx",
                ["Quy trình"],
            ),
            Block(
                "b2",
                "doc1",
                "table",
                rows_to_markdown(rows),
                "sample.xlsx",
                ["Biểu phí"],
                sheet_name="Biểu phí",
                cell_range="A1:B2",
                metadata={"rows": rows},
            ),
        ]

        parents, chunks = ParentChildChunker(config.chunking).chunk(document)

        self.assertEqual(2, len(parents))
        self.assertEqual(2, len(chunks))
        self.assertIn("Sheet: Biểu phí", chunks[1].content)
        self.assertEqual(36, len(qdrant_point_id(chunks[0].chunk_id)))

    def test_metadata_store_round_trip(self):
        store = MetadataStore(Path(":memory:"))
        document = ParsedDocument("doc1", "sample.pdf", "pdf", "hash")
        document.blocks = [Block("b1", "doc1", "text", "Nội dung", "sample.pdf", page=1)]
        parents, chunks = ParentChildChunker(PipelineConfig().chunking).chunk(document)

        store.upsert_document(document, parents, chunks)

        self.assertEqual(1, store.stats()["documents"])
        self.assertEqual(parents[0], store.get_parent(parents[0].parent_id))
        self.assertEqual(chunks[0], store.get_chunk(chunks[0].chunk_id))
        store.close()

    def test_ocr_result_text_collection(self):
        adapter = PaddleOCRVLAdapter("PaddleOCR-VL-1.6")
        payload = {
            "layout": [
                {"type": "text", "rec_texts": ["Dòng một", "Dòng hai"]},
                {"type": "table", "markdown": "| A | B |"},
            ]
        }
        collected = adapter._collect_text(payload)
        self.assertIn("Dòng một", collected)
        self.assertIn("| A | B |", collected)

    def test_vietnamese_tokenizer(self):
        tokens = tokenize_vi("Phí thường niên Visa Gold: 499.000 VNĐ")
        self.assertIn("phí", tokens)
        self.assertIn("visa", tokens)
        self.assertIn("499", tokens)


if __name__ == "__main__":
    unittest.main()
