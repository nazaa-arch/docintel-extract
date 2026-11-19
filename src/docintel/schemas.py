"""Strict Pydantic v2 schemas for document extraction, validation, and HITL review."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DocType = Literal["invoice", "purchase_order", "receipt"]
DocTypeRequest = Literal["auto", "invoice", "purchase_order", "receipt"]
JobStatus = Literal["APPROVED", "NEEDS_REVIEW", "REVIEWED"]


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class FieldConfidence(BaseModel):
    """Per-field extraction confidence, provenance, and review status."""

    field_name: str
    value: Any = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = Field(default="layout")
    needs_review: bool = False
    reason: str | None = None


class LineItem(BaseModel):
    """Structured financial line item."""

    model_config = ConfigDict(str_strip_whitespace=True)

    description: str = Field(min_length=1)
    quantity: float = Field(ge=0.0)
    unit_price: float = Field(ge=0.0)
    line_total: float = Field(ge=0.0)


class InvoiceData(BaseModel):
    """Structured schema for Indonesian and English tax/commercial invoices."""

    model_config = ConfigDict(str_strip_whitespace=True)

    invoice_number: str | None = None
    issue_date: str | None = None
    due_date: str | None = None
    vendor_name: str | None = None
    vendor_npwp: str | None = None
    buyer_name: str | None = None
    currency: str = "IDR"
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = None
    tax_amount: float | None = None
    total_amount: float | None = None


class PurchaseOrderData(BaseModel):
    """Structured schema for purchase orders."""

    model_config = ConfigDict(str_strip_whitespace=True)

    po_number: str | None = None
    order_date: str | None = None
    supplier_name: str | None = None
    buyer_name: str | None = None
    delivery_address: str | None = None
    currency: str = "IDR"
    line_items: list[LineItem] = Field(default_factory=list)
    total_amount: float | None = None


class ReceiptData(BaseModel):
    """Structured schema for point-of-sale and expense receipts."""

    model_config = ConfigDict(str_strip_whitespace=True)

    merchant_name: str | None = None
    transaction_date: str | None = None
    receipt_number: str | None = None
    payment_method: str | None = None
    currency: str = "IDR"
    line_items: list[LineItem] = Field(default_factory=list)
    tax_amount: float | None = None
    total_amount: float | None = None


DocumentData = InvoiceData | PurchaseOrderData | ReceiptData


def schema_for_doc_type(doc_type: DocType) -> type[DocumentData]:
    """Return the Pydantic schema class corresponding to a document type."""
    mapping: dict[DocType, type[DocumentData]] = {
        "invoice": InvoiceData,
        "purchase_order": PurchaseOrderData,
        "receipt": ReceiptData,
    }
    return mapping[doc_type]


class ExtractionResult(BaseModel):
    """Complete extraction, reconciliation, and HITL triage result."""

    job_id: str
    filename: str | None = None
    doc_type: DocType
    status: JobStatus
    provider: str = "local"
    overall_confidence: float = Field(ge=0.0, le=1.0)
    reconciliation_passed: bool
    reconciliation_notes: list[str] = Field(default_factory=list)
    data: InvoiceData | PurchaseOrderData | ReceiptData
    field_confidences: list[FieldConfidence] = Field(default_factory=list)
    latency_ms: float = Field(ge=0.0)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str | None = None
    reviewer_notes: str | None = None


class TextExtractRequest(BaseModel):
    """JSON payload for direct text document extraction."""

    text: str = Field(min_length=1)
    filename: str = "document.txt"
    doc_type: DocTypeRequest = "auto"
    provider: str | None = None


class BatchTextExtractRequest(BaseModel):
    """JSON payload for batch text extraction."""

    documents: list[TextExtractRequest] = Field(min_length=1)


class BatchExtractResponse(BaseModel):
    """Batch extraction summary and per-document results."""

    total: int
    approved: int
    needs_review: int
    results: list[ExtractionResult]


class ReviewSubmission(BaseModel):
    """Human-in-the-Loop correction and approval submission."""

    data: dict[str, Any]
    reviewer_notes: str | None = None
    approve: bool = True


class HealthResponse(BaseModel):
    """Service health and queue telemetry."""

    status: str
    version: str
    provider: str
    review_confidence_threshold: float
    jobs_summary: dict[str, int]
