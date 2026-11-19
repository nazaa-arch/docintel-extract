"""DocIntel Extract: Multimodal Document Intelligence Pipeline."""

__version__ = "1.0.0"

from docintel.config import Settings, get_settings
from docintel.schemas import (
    DocType,
    ExtractionResult,
    FieldConfidence,
    InvoiceData,
    JobStatus,
    LineItem,
    PurchaseOrderData,
    ReceiptData,
)

__all__ = [
    "__version__",
    "DocType",
    "ExtractionResult",
    "FieldConfidence",
    "InvoiceData",
    "JobStatus",
    "LineItem",
    "PurchaseOrderData",
    "ReceiptData",
    "Settings",
    "get_settings",
]
