"""Unit tests for Indonesian/English parsing, document classification, and LLM adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from docintel.config import Settings
from docintel.extractor import DocumentExtractor, classify_document, parse_date, parse_money
from docintel.ingestion import ingest_file, ingest_text
from docintel.schemas import InvoiceData, PurchaseOrderData, ReceiptData

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Rp 1.250.000", 1250000.0),
        ("Rp1.250.000,50", 1250000.5),
        ("Rp 45.000,-", 45000.0),
        ("IDR 13.875.000", 13875000.0),
        ("$1,250.00", 1250.0),
        ("USD 49.95", 49.95),
        ("SGD 3,400.00", 3400.0),
        ("11%", None),
    ],
)
def test_parse_money_id_and_en(raw: str, expected: float | None) -> None:
    assert parse_money(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-03-15", "2026-03-15"),
        ("15 Maret 2026", "2026-03-15"),
        ("02 Februari 2026", "2026-02-02"),
        ("20 Mei 2026", "2026-05-20"),
        ("14 Okt 2026", "2026-10-14"),
        ("05 Desember 2026", "2026-12-05"),
        ("March 10, 2026", "2026-03-10"),
        ("15/04/2026", "2026-04-15"),
        ("TBD", None),
    ],
)
def test_parse_date_id_and_en(raw: str, expected: str | None) -> None:
    assert parse_date(raw) == expected


def test_extract_indonesian_invoice_pdf(test_settings: Settings) -> None:
    doc = ingest_file(SAMPLES_DIR / "invoice_id_01.pdf")
    assert classify_document(doc) == "invoice"

    out = DocumentExtractor(test_settings).extract(doc, doc_type="auto")
    assert out.doc_type == "invoice"
    assert isinstance(out.data, InvoiceData)
    assert out.data.invoice_number == "INV-2026-0301"
    assert out.data.vendor_npwp == "01.234.567.8-012.000"
    assert out.data.issue_date == "2026-03-15"
    assert out.data.due_date == "2026-04-14"
    assert out.data.currency == "IDR"
    assert len(out.data.line_items) == 2
    assert out.data.subtotal == 12500000.0
    assert out.data.tax_amount == 1375000.0
    assert out.data.total_amount == 13875000.0


def test_extract_purchase_order_pdf(test_settings: Settings) -> None:
    doc = ingest_file(SAMPLES_DIR / "po_en_01.pdf")
    out = DocumentExtractor(test_settings).extract(doc, doc_type="auto")

    assert out.doc_type == "purchase_order"
    assert isinstance(out.data, PurchaseOrderData)
    assert out.data.po_number == "PO-EN-7701"
    assert out.data.currency == "USD"
    assert out.data.total_amount == 4440.0
    assert len(out.data.line_items) == 2


def test_extract_receipt_pdf(test_settings: Settings) -> None:
    doc = ingest_file(SAMPLES_DIR / "receipt_id_01.pdf")
    out = DocumentExtractor(test_settings).extract(doc, doc_type="auto")

    assert out.doc_type == "receipt"
    assert isinstance(out.data, ReceiptData)
    assert out.data.merchant_name == "Kopi Kenangan Senopati"
    assert out.data.receipt_number == "STR-2026-8812"
    assert out.data.payment_method == "QRIS BCA"
    assert out.data.total_amount == 121000.0


def test_extract_receipt_from_plain_text_x_pattern(test_settings: Settings) -> None:
    text = (
        "STRUK PEMBAYARAN\n"
        "Merchant: Kedai Kopi Nusantara\n"
        "No. Struk: STR-2026-501\n"
        "Tanggal Transaksi: 19 Mei 2026\n"
        "Metode Pembayaran: QRIS\n"
        "2 x Es Kopi Susu @ Rp 22.000  Rp 44.000\n"
        "1 x Roti Bakar @ Rp 18.000  Rp 18.000\n"
        "PB1 10%: Rp 6.200\n"
        "Total Bayar: Rp 68.200\n"
    )
    doc = ingest_text(text, filename="struk.txt")
    out = DocumentExtractor(test_settings).extract(doc, doc_type="auto")
    assert out.doc_type == "receipt"
    assert isinstance(out.data, ReceiptData)
    assert len(out.data.line_items) == 2
    assert out.data.total_amount == 68200.0


def test_ollama_adapter_parses_structured_json(monkeypatch: pytest.MonkeyPatch, test_settings: Settings) -> None:
    mock_payload = {
        "invoice_number": "INV-LLM-01",
        "issue_date": "2026-06-01",
        "due_date": "2026-07-01",
        "vendor_name": "PT Ollama Vision AI",
        "vendor_npwp": "01.111.222.3-444.000",
        "buyer_name": "PT Client Test",
        "currency": "IDR",
        "line_items": [
            {"description": "Vision Extraction Node", "quantity": 1, "unit_price": 1000000, "line_total": 1000000}
        ],
        "subtotal": 1000000,
        "tax_amount": 110000,
        "total_amount": 1110000,
    }

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"message": {"content": f"```json\n{json.dumps(mock_payload)}\n```"}}

    def fake_post(self: httpx.Client, url: str, **kwargs: Any) -> FakeResponse:
        assert url.endswith("/api/chat")
        assert kwargs["json"]["format"] == "json"
        return FakeResponse()

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    doc = ingest_text("INVOICE\nVendor: PT Ollama Vision AI", filename="llm.txt")
    out = DocumentExtractor(test_settings).extract(doc, doc_type="invoice", provider_override="ollama")

    assert out.provider == "ollama"
    assert isinstance(out.data, InvoiceData)
    assert out.data.invoice_number == "INV-LLM-01"
    assert out.data.total_amount == 1110000.0


def test_cloud_adapters_require_api_keys(test_settings: Settings) -> None:
    doc = ingest_text("INVOICE #1", filename="a.txt")
    extractor = DocumentExtractor(test_settings)
    with pytest.raises(ValueError, match="DOCINTEL_GEMINI_API_KEY"):
        extractor.extract(doc, doc_type="invoice", provider_override="gemini")
    with pytest.raises(ValueError, match="DOCINTEL_OPENAI_API_KEY"):
        extractor.extract(doc, doc_type="invoice", provider_override="openai")


def test_gemini_adapter_uses_header_auth_and_inline_image(monkeypatch: pytest.MonkeyPatch, test_settings: Settings) -> None:
    settings = test_settings.model_copy(update={"gemini_api_key": "test-gem-secret", "model": "gemini-2.0-flash"})
    mock_payload = {
        "invoice_number": "INV-GEM-01",
        "issue_date": "2026-06-01",
        "due_date": "2026-07-01",
        "vendor_name": "PT Gemini Vision",
        "vendor_npwp": None,
        "buyer_name": "PT Client",
        "currency": "USD",
        "line_items": [{"description": "API Credits", "quantity": 1, "unit_price": 100.0, "line_total": 100.0}],
        "subtotal": 100.0,
        "tax_amount": 10.0,
        "total_amount": 110.0,
    }

    class FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"candidates": [{"content": {"parts": [{"text": json.dumps(mock_payload)}]}}]}

    def fake_post(self: httpx.Client, url: str, **kwargs: Any) -> FakeResp:
        assert "generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent" in url
        # API key must be sent in header, never exposed in URL query params
        assert "params" not in kwargs or not kwargs["params"]
        assert kwargs["headers"]["x-goog-api-key"] == "test-gem-secret"
        parts = kwargs["json"]["contents"][0]["parts"]
        assert any("inlineData" in p for p in parts)
        return FakeResp()

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    doc = ingest_file(SAMPLES_DIR / "invoice_en_01.pdf", render_images=True)
    out = DocumentExtractor(settings).extract(doc, doc_type="invoice", provider_override="gemini")
    assert out.provider == "gemini"
    assert isinstance(out.data, InvoiceData)
    assert out.data.invoice_number == "INV-GEM-01"


def test_openai_adapter_payload_and_json_parsing(monkeypatch: pytest.MonkeyPatch, test_settings: Settings) -> None:
    settings = test_settings.model_copy(update={"openai_api_key": "test-oai-secret", "model": "gpt-4o-mini"})
    mock_payload = {
        "merchant_name": "Vision Cafe",
        "transaction_date": "2026-06-05",
        "receipt_number": "RCPT-OAI-1",
        "payment_method": "Card",
        "currency": "USD",
        "line_items": [{"description": "Espresso", "quantity": 2, "unit_price": 4.0, "line_total": 8.0}],
        "tax_amount": 0.8,
        "total_amount": 8.8,
    }

    class FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": f"Here is the JSON:\n{json.dumps(mock_payload)}"}}]}

    def fake_post(self: httpx.Client, url: str, **kwargs: Any) -> FakeResp:
        assert url.endswith("/chat/completions")
        assert kwargs["headers"]["Authorization"] == "Bearer test-oai-secret"
        assert kwargs["json"]["response_format"] == {"type": "json_object"}
        content = kwargs["json"]["messages"][0]["content"]
        assert any(part.get("type") == "image_url" for part in content)
        return FakeResp()

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    doc = ingest_file(SAMPLES_DIR / "receipt_en_01.pdf", render_images=True)
    out = DocumentExtractor(settings).extract(doc, doc_type="receipt", provider_override="openai")
    assert out.provider == "openai"
    assert isinstance(out.data, ReceiptData)
    assert out.data.receipt_number == "RCPT-OAI-1"
