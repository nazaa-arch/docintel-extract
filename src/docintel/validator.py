"""Per-field confidence scoring, cross-field arithmetic reconciliation, and HITL triage."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from docintel.schemas import (
    DocType,
    DocumentData,
    FieldConfidence,
    InvoiceData,
    JobStatus,
    LineItem,
    PurchaseOrderData,
    ReceiptData,
)

VALID_CURRENCIES = {"IDR", "USD", "EUR", "SGD", "GBP", "JPY", "MYR", "AUD"}
ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NPWP_VALID_PATTERN = re.compile(r"^(?:\d{2}\.\d{3}\.\d{3}\.\d-\d{3}\.\d{3}|\d{4}\.\d{4}\.\d{4}\.\d{4}|\d{16})$")

REQUIRED_FIELDS: dict[DocType, set[str]] = {
    "invoice": {
        "invoice_number",
        "issue_date",
        "vendor_name",
        "buyer_name",
        "currency",
        "line_items",
        "subtotal",
        "total_amount",
    },
    "purchase_order": {
        "po_number",
        "order_date",
        "supplier_name",
        "buyer_name",
        "delivery_address",
        "currency",
        "line_items",
        "total_amount",
    },
    "receipt": {
        "merchant_name",
        "transaction_date",
        "receipt_number",
        "payment_method",
        "currency",
        "line_items",
        "total_amount",
    },
}

SOURCE_BONUS: dict[str, float] = {
    "human": 0.08,
    "table": 0.03,
    "layout_table": 0.02,
    "label": 0.02,
    "summary": 0.02,
    "regex": 0.01,
    "header": -0.01,
    "label_next_line": 0.0,
    "llm:ollama": 0.0,
    "llm:gemini": 0.01,
    "llm:openai": 0.01,
}


@dataclass
class ValidationOutcome:
    """Validation & triage summary returned by validate_extraction."""

    status: JobStatus
    overall_confidence: float
    reconciliation_passed: bool
    reconciliation_notes: list[str]
    field_confidences: list[FieldConfidence]


def _currency_tolerance(currency: str) -> float:
    return 1.0 if currency.upper() in {"IDR", "JPY", "KRW"} else 0.02


def _is_valid_iso_date(value: str | None) -> bool:
    if not value or not ISO_DATE_PATTERN.match(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _clamp(score: float) -> float:
    return round(max(0.0, min(1.0, score)), 4)


def reconcile_arithmetic(
    doc_type: DocType,
    data: DocumentData,
) -> tuple[bool, list[str], set[str]]:
    """Verify line-item multiplication, subtotal summation, tax + subtotal == total, and date order.

    Returns:
        (reconciliation_passed, reconciliation_notes, failed_fields)
    """
    notes: list[str] = []
    failed_fields: set[str] = set()
    passed = True

    tol = _currency_tolerance(data.currency)
    items: list[LineItem] = list(data.line_items)

    if not items:
        passed = False
        failed_fields.add("line_items")
        notes.append("FAIL: No line items extracted; cannot reconcile line totals.")
    else:
        line_errors = 0
        for idx, item in enumerate(items, start=1):
            expected_line = round(item.quantity * item.unit_price, 2)
            diff = abs(expected_line - item.line_total)
            if diff > tol:
                passed = False
                line_errors += 1
                failed_fields.add("line_items")
                notes.append(
                    f"FAIL: Line item #{idx} ('{item.description}') mismatch: "
                    f"{item.quantity:g} * {item.unit_price:,.2f} = {expected_line:,.2f} != {item.line_total:,.2f}"
                )
        if line_errors == 0:
            notes.append(f"PASS: All {len(items)} line item(s) satisfy qty * unit_price == line_total.")

    items_sum = round(sum(item.line_total for item in items), 2)

    if isinstance(data, InvoiceData):
        if data.subtotal is None:
            passed = False
            failed_fields.add("subtotal")
            notes.append("FAIL: Invoice subtotal is missing.")
        elif items:
            sub_diff = abs(items_sum - data.subtotal)
            if sub_diff > tol:
                passed = False
                failed_fields.update({"subtotal", "line_items"})
                notes.append(
                    f"FAIL: Sum of line items ({items_sum:,.2f}) != subtotal ({data.subtotal:,.2f}); "
                    f"delta = {sub_diff:,.2f}."
                )
            else:
                notes.append(f"PASS: Sum of line items ({items_sum:,.2f}) matches subtotal ({data.subtotal:,.2f}).")

        if data.total_amount is None:
            passed = False
            failed_fields.add("total_amount")
            notes.append("FAIL: Invoice total_amount is missing.")
        elif data.subtotal is not None:
            tax = data.tax_amount or 0.0
            expected_total = round(data.subtotal + tax, 2)
            total_diff = abs(expected_total - data.total_amount)
            if total_diff > tol:
                passed = False
                failed_fields.update({"total_amount", "subtotal"})
                if data.tax_amount is not None:
                    failed_fields.add("tax_amount")
                notes.append(
                    f"FAIL: subtotal ({data.subtotal:,.2f}) + tax ({tax:,.2f}) = {expected_total:,.2f} "
                    f"!= total_amount ({data.total_amount:,.2f}); delta = {total_diff:,.2f}."
                )
            else:
                notes.append(
                    f"PASS: subtotal ({data.subtotal:,.2f}) + tax ({tax:,.2f}) == total_amount ({data.total_amount:,.2f})."
                )

        if _is_valid_iso_date(data.issue_date) and _is_valid_iso_date(data.due_date):
            if str(data.due_date) < str(data.issue_date):
                passed = False
                failed_fields.add("due_date")
                notes.append(f"FAIL: due_date ({data.due_date}) precedes issue_date ({data.issue_date}).")
            else:
                notes.append(f"PASS: due_date ({data.due_date}) is on or after issue_date ({data.issue_date}).")

    elif isinstance(data, PurchaseOrderData):
        if data.total_amount is None:
            passed = False
            failed_fields.add("total_amount")
            notes.append("FAIL: Purchase order total_amount is missing.")
        elif items:
            po_diff = abs(items_sum - data.total_amount)
            if po_diff > tol:
                passed = False
                failed_fields.update({"total_amount", "line_items"})
                notes.append(
                    f"FAIL: Sum of PO line items ({items_sum:,.2f}) != total_amount ({data.total_amount:,.2f}); "
                    f"delta = {po_diff:,.2f}."
                )
            else:
                notes.append(f"PASS: Sum of PO line items ({items_sum:,.2f}) matches total_amount ({data.total_amount:,.2f}).")

    elif isinstance(data, ReceiptData):
        if data.total_amount is None:
            passed = False
            failed_fields.add("total_amount")
            notes.append("FAIL: Receipt total_amount is missing.")
        elif items:
            tax = data.tax_amount or 0.0
            expected_total = round(items_sum + tax, 2)
            rcpt_diff = abs(expected_total - data.total_amount)
            if rcpt_diff > tol:
                passed = False
                failed_fields.update({"total_amount", "line_items"})
                if data.tax_amount is not None:
                    failed_fields.add("tax_amount")
                notes.append(
                    f"FAIL: Sum of receipt items ({items_sum:,.2f}) + tax ({tax:,.2f}) = {expected_total:,.2f} "
                    f"!= total_amount ({data.total_amount:,.2f}); delta = {rcpt_diff:,.2f}."
                )
            else:
                notes.append(
                    f"PASS: Sum of receipt items ({items_sum:,.2f}) + tax ({tax:,.2f}) == total_amount ({data.total_amount:,.2f})."
                )

    return passed, notes, failed_fields


def _score_field(
    field_name: str,
    value: Any,
    doc_type: DocType,
    currency: str,
    source: str,
    reconciliation_passed: bool,
    failed_fields: set[str],
) -> FieldConfidence:
    """Compute calibrated confidence score [0.0, 1.0] for an individual extracted field."""
    required = field_name in REQUIRED_FIELDS[doc_type]
    bonus = SOURCE_BONUS.get(source, 0.0)

    # Human-verified non-empty fields with no reconciliation failure get near-perfect confidence
    if source == "human" and field_name not in failed_fields and value not in (None, "", []):
        return FieldConfidence(
            field_name=field_name,
            value=value if field_name != "line_items" else f"{len(value)} item(s)",
            confidence=0.99,
            source="human",
            needs_review=False,
            reason=None,
        )

    # Missing value handling
    if value is None or value == "" or (isinstance(value, list) and len(value) == 0):
        if required:
            return FieldConfidence(
                field_name=field_name,
                value=None,
                confidence=0.10,
                source=source or "missing",
                needs_review=True,
                reason=f"Missing required field '{field_name}'",
            )
        # Optional field: vendor_npwp is only relevant for IDR invoices; tax_amount optional on tax-free receipts
        if field_name == "vendor_npwp" and currency != "IDR":
            return FieldConfidence(
                field_name=field_name,
                value=None,
                confidence=0.90,
                source="optional",
                needs_review=False,
                reason=None,
            )
        if field_name == "tax_amount":
            return FieldConfidence(
                field_name=field_name,
                value=0.0,
                confidence=0.88,
                source="optional",
                needs_review=False,
                reason=None,
            )
        return FieldConfidence(
            field_name=field_name,
            value=None,
            confidence=0.78,
            source=source or "missing",
            needs_review=False,
            reason=f"Optional field '{field_name}' not present",
        )

    # Type-specific semantic validation
    if field_name == "line_items":
        item_count = len(value)
        if field_name in failed_fields:
            return FieldConfidence(
                field_name=field_name,
                value=f"{item_count} item(s)",
                confidence=0.42,
                source=source,
                needs_review=True,
                reason="Arithmetic mismatch in line items or subtotal",
            )
        base = 0.93 + bonus
        return FieldConfidence(
            field_name=field_name,
            value=f"{item_count} item(s)",
            confidence=_clamp(base),
            source=source,
            needs_review=False,
            reason=None,
        )

    if field_name.endswith("_date"):
        if _is_valid_iso_date(str(value)):
            if field_name in failed_fields:
                return FieldConfidence(
                    field_name=field_name,
                    value=value,
                    confidence=0.45,
                    source=source,
                    needs_review=True,
                    reason="Date chronology error (due_date precedes issue_date)",
                )
            return FieldConfidence(
                field_name=field_name,
                value=value,
                confidence=_clamp(0.94 + bonus),
                source=source,
                needs_review=False,
                reason=None,
            )
        return FieldConfidence(
            field_name=field_name,
            value=value,
            confidence=0.45,
            source=source,
            needs_review=True,
            reason=f"Unparseable date format: '{value}'",
        )

    if field_name == "vendor_npwp":
        if NPWP_VALID_PATTERN.fullmatch(str(value).strip()):
            return FieldConfidence(
                field_name=field_name,
                value=value,
                confidence=_clamp(0.95 + bonus),
                source=source,
                needs_review=False,
                reason=None,
            )
        return FieldConfidence(
            field_name=field_name,
            value=value,
            confidence=0.48,
            source=source,
            needs_review=True,
            reason=f"Invalid Indonesian NPWP format: '{value}'",
        )

    if field_name == "currency":
        if str(value).upper() in VALID_CURRENCIES:
            return FieldConfidence(
                field_name=field_name,
                value=str(value).upper(),
                confidence=0.97,
                source=source,
                needs_review=False,
                reason=None,
            )
        return FieldConfidence(
            field_name=field_name,
            value=value,
            confidence=0.55,
            source=source,
            needs_review=True,
            reason=f"Unrecognized currency code: '{value}'",
        )

    if field_name in {"subtotal", "tax_amount", "total_amount"}:
        num = float(value)
        if num < 0:
            return FieldConfidence(
                field_name=field_name,
                value=num,
                confidence=0.35,
                source=source,
                needs_review=True,
                reason="Negative monetary amount",
            )
        if field_name in failed_fields:
            return FieldConfidence(
                field_name=field_name,
                value=num,
                confidence=0.38,
                source=source,
                needs_review=True,
                reason="Failed cross-field arithmetic reconciliation",
            )
        base = 0.95 if reconciliation_passed else 0.84
        return FieldConfidence(
            field_name=field_name,
            value=num,
            confidence=_clamp(base + bonus),
            source=source,
            needs_review=False,
            reason=None,
        )

    if field_name in {"invoice_number", "po_number", "receipt_number"}:
        val_s = str(value).strip()
        if len(val_s) >= 4 and re.search(r"\d", val_s):
            return FieldConfidence(
                field_name=field_name,
                value=val_s,
                confidence=_clamp(0.93 + bonus),
                source=source,
                needs_review=False,
                reason=None,
            )
        return FieldConfidence(
            field_name=field_name,
            value=val_s,
            confidence=0.52,
            source=source,
            needs_review=True,
            reason="Document identifier looks incomplete or malformed",
        )

    # Generic text fields (vendor_name, buyer_name, supplier_name, merchant_name, delivery_address, payment_method)
    val_s = str(value).strip()
    if len(val_s) >= 3 and re.search(r"[A-Za-z]", val_s):
        return FieldConfidence(
            field_name=field_name,
            value=val_s,
            confidence=_clamp(0.91 + bonus),
            source=source,
            needs_review=False,
            reason=None,
        )
    return FieldConfidence(
        field_name=field_name,
        value=val_s,
        confidence=0.55,
        source=source,
        needs_review=True,
        reason=f"Extracted text for '{field_name}' is suspiciously short",
    )


def validate_extraction(
    doc_type: DocType,
    data: DocumentData,
    field_sources: dict[str, str] | None = None,
    review_confidence_threshold: float = 0.80,
) -> ValidationOutcome:
    """Run arithmetic reconciliation, per-field confidence scoring, and HITL triage."""
    sources = field_sources or {}
    reconciliation_passed, reconciliation_notes, failed_fields = reconcile_arithmetic(doc_type, data)

    field_confidences: list[FieldConfidence] = []
    weighted_sum = 0.0
    total_weight = 0.0
    any_flagged = False

    for field_name in type(data).model_fields:
        value = getattr(data, field_name)
        src = sources.get(field_name, "layout")
        fc = _score_field(
            field_name=field_name,
            value=value,
            doc_type=doc_type,
            currency=data.currency,
            source=src,
            reconciliation_passed=reconciliation_passed,
            failed_fields=failed_fields,
        )
        field_confidences.append(fc)

        is_required = field_name in REQUIRED_FIELDS[doc_type]
        weight = 2.0 if field_name in {"total_amount", "line_items", "subtotal"} else (1.4 if is_required else 0.6)
        weighted_sum += fc.confidence * weight
        total_weight += weight
        if fc.needs_review:
            any_flagged = True

    overall_confidence = _clamp(weighted_sum / total_weight) if total_weight > 0 else 0.0

    # Apply reconciliation penalty to overall score if cross-field math failed
    if not reconciliation_passed:
        overall_confidence = _clamp(min(overall_confidence, 0.72))

    needs_review = (
        overall_confidence < review_confidence_threshold
        or not reconciliation_passed
        or any_flagged
    )
    status: JobStatus = "NEEDS_REVIEW" if needs_review else "APPROVED"

    return ValidationOutcome(
        status=status,
        overall_confidence=overall_confidence,
        reconciliation_passed=reconciliation_passed,
        reconciliation_notes=reconciliation_notes,
        field_confidences=field_confidences,
    )
