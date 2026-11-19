"""Unit tests for PDF, image, and text ingestion."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from docintel.ingestion import ingest_bytes, ingest_file, ingest_image_bytes, ingest_text

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


def test_ingest_bordered_pdf_extracts_blocks_and_tables() -> None:
    pdf_path = SAMPLES_DIR / "invoice_id_01.pdf"
    doc = ingest_file(pdf_path, render_images=True)

    assert doc.source_format == "pdf"
    assert doc.page_count == 1
    assert "PT Nusantara Cloud Teknika" in doc.raw_text
    assert "INV-2026-0301" in doc.raw_text
    assert len(doc.blocks) >= 3
    assert len(doc.tables) >= 1
    assert len(doc.page_images_b64) == 1


def test_ingest_borderless_pdf_reconstructs_columns() -> None:
    pdf_path = SAMPLES_DIR / "po_id_02.pdf"
    doc = ingest_file(pdf_path, render_images=False)

    assert doc.source_format == "pdf"
    assert "PO-2026-0244" in doc.raw_text
    assert len(doc.tables) >= 1
    assert doc.page_images_b64 == []


def test_ingest_image_pillow_metadata_and_base64() -> None:
    img = Image.new("RGB", (320, 180), color=(24, 34, 52))
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("docintel_text", "SALES RECEIPT\nMerchant: Pixel Coffee")
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=pnginfo)

    doc = ingest_image_bytes(buf.getvalue(), filename="receipt_scan.png")
    assert doc.source_format == "image"
    assert doc.image_metadata is not None
    assert doc.image_metadata["width"] == 320
    assert doc.image_metadata["height"] == 180
    assert doc.image_metadata["format"] == "PNG"
    assert len(doc.page_images_b64) == 1
    assert "Pixel Coffee" in doc.raw_text


def test_ingest_text_pipe_table() -> None:
    raw = (
        "COMMERCIAL INVOICE\n"
        "Invoice No: INV-2026-100\n"
        "| Description | Qty | Unit Price | Line Total |\n"
        "| --- | --- | --- | --- |\n"
        "| Cloud GPU | 2 | $500.00 | $1,000.00 |\n"
    )
    doc = ingest_text(raw, filename="invoice.txt")
    assert doc.source_format == "text"
    assert len(doc.tables) == 1
    assert len(doc.tables[0]) == 2


def test_ingest_bytes_rejects_unsupported_extension() -> None:
    with pytest.raises(ValueError, match="Unsupported file format"):
        ingest_bytes(b"\x00\x01\x02", filename="archive.zip", content_type="application/zip")
