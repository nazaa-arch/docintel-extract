"""Unit tests for confidence scoring, arithmetic reconciliation, and HITL triage."""

from __future__ import annotations

from docintel.schemas import InvoiceData, LineItem, PurchaseOrderData, ReceiptData
from docintel.validator import validate_extraction


def test_clean_invoice_reconciles_and_approves() -> None:
    invoice = InvoiceData(
        invoice_number="INV-2026-1001",
        issue_date="2026-03-01",
        due_date="2026-03-31",
        vendor_name="PT Tekno Andal",
        vendor_npwp="01.234.567.8-012.000",
        buyer_name="PT Pembeli Utama",
        currency="IDR",
        line_items=[
            LineItem(description="GPU Server", quantity=2, unit_price=5000000, line_total=10000000),
        ],
        subtotal=10000000,
        tax_amount=1100000,
        total_amount=11100000,
    )
    outcome = validate_extraction("invoice", invoice, review_confidence_threshold=0.80)
    assert outcome.reconciliation_passed is True
    assert outcome.status == "APPROVED"
    assert outcome.overall_confidence >= 0.90


def test_invoice_total_mismatch_triggers_needs_review() -> None:
    invoice = InvoiceData(
        invoice_number="INV-2026-1002",
        issue_date="2026-03-01",
        due_date="2026-03-31",
        vendor_name="PT Tekno Andal",
        vendor_npwp="01.234.567.8-012.000",
        buyer_name="PT Pembeli Utama",
        currency="IDR",
        line_items=[
            LineItem(description="GPU Server", quantity=2, unit_price=5000000, line_total=10000000),
        ],
        subtotal=10000000,
        tax_amount=1100000,
        total_amount=11500000,  # Wrong total (+400,000)
    )
    outcome = validate_extraction("invoice", invoice, review_confidence_threshold=0.80)
    assert outcome.reconciliation_passed is False
    assert outcome.status == "NEEDS_REVIEW"
    assert outcome.overall_confidence <= 0.72
    assert any("FAIL:" in note for note in outcome.reconciliation_notes)


def test_po_line_item_multiplication_mismatch_flagged() -> None:
    po = PurchaseOrderData(
        po_number="PO-2026-55",
        order_date="2026-04-01",
        supplier_name="Supplier Corp",
        buyer_name="Buyer Corp",
        delivery_address="Jl. Sudirman 10, Jakarta",
        currency="IDR",
        line_items=[
            LineItem(description="Chair", quantity=5, unit_price=450000, line_total=2500000),
        ],
        total_amount=2500000,
    )
    outcome = validate_extraction("purchase_order", po)
    assert outcome.reconciliation_passed is False
    assert outcome.status == "NEEDS_REVIEW"
    item_fc = next(fc for fc in outcome.field_confidences if fc.field_name == "line_items")
    assert item_fc.needs_review is True


def test_missing_required_fields_and_bad_npwp_flagged() -> None:
    invoice = InvoiceData(
        invoice_number=None,
        issue_date="NOT-A-DATE",
        due_date="2026-04-01",
        vendor_name="PT Contoh",
        vendor_npwp="12345-BAD",
        buyer_name="PT Klien",
        currency="IDR",
        line_items=[LineItem(description="Item A", quantity=1, unit_price=100000, line_total=100000)],
        subtotal=100000,
        tax_amount=11000,
        total_amount=111000,
    )
    outcome = validate_extraction("invoice", invoice)
    assert outcome.status == "NEEDS_REVIEW"
    flagged = {fc.field_name for fc in outcome.field_confidences if fc.needs_review}
    assert {"invoice_number", "issue_date", "vendor_npwp"} <= flagged


def test_receipt_reconciliation_detects_tax_plus_items_error() -> None:
    rcpt = ReceiptData(
        merchant_name="Roasters Cafe",
        transaction_date="2026-05-01",
        receipt_number="RCPT-1029",
        payment_method="Visa",
        currency="USD",
        line_items=[LineItem(description="Latte", quantity=2, unit_price=5.0, line_total=10.0)],
        tax_amount=1.0,
        total_amount=15.0,
    )
    outcome = validate_extraction("receipt", rcpt)
    assert outcome.reconciliation_passed is False
    assert outcome.status == "NEEDS_REVIEW"
