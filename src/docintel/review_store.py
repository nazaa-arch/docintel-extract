"""SQLite persistence layer for the HITL review queue and batch JSON/CSV export."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from docintel.schemas import (
    DocType,
    ExtractionResult,
    FieldConfidence,
    InvoiceData,
    PurchaseOrderData,
    ReceiptData,
    schema_for_doc_type,
)

ExportFormat = Literal["json", "csv"]
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _sanitize_csv_cell(value: Any) -> Any:
    """Neutralize spreadsheet formula injection (OWASP CSV/DDE injection) on text cells."""
    if isinstance(value, str):
        stripped = value.lstrip(" ")
        if stripped and stripped[0] in _CSV_FORMULA_PREFIXES:
            return f"'{value}"
    return value


class ReviewStore:
    """Thread-safe SQLite store for extraction jobs and Human-in-the-Loop review workflows."""

    def __init__(self, db_path: Path, auto_init: bool = True) -> None:
        self.db_path = Path(db_path)
        self._initialized = False
        if auto_init:
            self.init_schema()

    def init_schema(self) -> None:
        """Create parent directories, enable WAL mode, and initialize SQLite schema once."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect(ensure_init=False) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS extraction_jobs (
                    job_id TEXT PRIMARY KEY,
                    filename TEXT,
                    doc_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    overall_confidence REAL NOT NULL,
                    reconciliation_passed INTEGER NOT NULL,
                    reconciliation_notes_json TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    field_confidences_json TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    reviewer_notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON extraction_jobs(status, created_at DESC)"
            )
        self._initialized = True

    @contextmanager
    def _connect(self, ensure_init: bool = True) -> Iterator[sqlite3.Connection]:
        if ensure_init and not self._initialized:
            self.init_schema()
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=15000;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _row_to_result(self, row: sqlite3.Row) -> ExtractionResult:
        doc_type: DocType = row["doc_type"]
        schema_cls = schema_for_doc_type(doc_type)
        data = schema_cls.model_validate_json(row["data_json"])
        field_confidences = [
            FieldConfidence.model_validate(item) for item in json.loads(row["field_confidences_json"])
        ]
        reconciliation_notes = json.loads(row["reconciliation_notes_json"])
        return ExtractionResult(
            job_id=row["job_id"],
            filename=row["filename"],
            doc_type=doc_type,
            status=row["status"],
            provider=row["provider"],
            overall_confidence=float(row["overall_confidence"]),
            reconciliation_passed=bool(row["reconciliation_passed"]),
            reconciliation_notes=reconciliation_notes,
            data=data,
            field_confidences=field_confidences,
            latency_ms=float(row["latency_ms"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            reviewer_notes=row["reviewer_notes"],
        )

    def save_job(self, result: ExtractionResult) -> ExtractionResult:
        """Insert or replace an extraction job in SQLite."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO extraction_jobs (
                    job_id, filename, doc_type, status, provider,
                    overall_confidence, reconciliation_passed, reconciliation_notes_json,
                    data_json, field_confidences_json, latency_ms,
                    reviewer_notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.job_id,
                    result.filename,
                    result.doc_type,
                    result.status,
                    result.provider,
                    result.overall_confidence,
                    1 if result.reconciliation_passed else 0,
                    json.dumps(result.reconciliation_notes, ensure_ascii=False),
                    result.data.model_dump_json(),
                    json.dumps([fc.model_dump() for fc in result.field_confidences], ensure_ascii=False),
                    result.latency_ms,
                    result.reviewer_notes,
                    result.created_at,
                    result.updated_at,
                ),
            )
        return result

    def update_job(self, result: ExtractionResult) -> ExtractionResult:
        """Persist an updated extraction job after HITL review."""
        return self.save_job(result)

    def get_job(self, job_id: str) -> ExtractionResult | None:
        """Retrieve a single job by ID."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM extraction_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_result(row) if row else None

    def list_jobs(
        self,
        status: str | None = None,
        doc_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ExtractionResult]:
        """List extraction jobs optionally filtered by status and/or doc_type."""
        clauses: list[str] = []
        params: list[object] = []

        if status:
            clauses.append("status = ?")
            params.append(status.upper())
        if doc_type:
            clauses.append("doc_type = ?")
            params.append(doc_type.lower())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM extraction_jobs {where_sql} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_result(row) for row in rows]

    def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM extraction_jobs WHERE job_id = ?", (job_id,))
        return cur.rowcount > 0

    def get_status_counts(self) -> dict[str, int]:
        """Return queue counts grouped by job status."""
        counts = {"APPROVED": 0, "NEEDS_REVIEW": 0, "REVIEWED": 0, "TOTAL": 0}
        with self._connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS cnt FROM extraction_jobs GROUP BY status").fetchall()
        for row in rows:
            status = row["status"]
            cnt = int(row["cnt"])
            counts[status] = cnt
            counts["TOTAL"] += cnt
        return counts

    def export_records(
        self,
        fmt: ExportFormat = "json",
        statuses: list[str] | None = None,
    ) -> tuple[str, str]:
        """Export approved/reviewed (or specified) extraction jobs as JSON or flattened CSV."""
        allowed = [s.upper() for s in (["APPROVED", "REVIEWED"] if statuses is None else statuses)]
        if not allowed:
            results: list[ExtractionResult] = []
        else:
            placeholders = ", ".join("?" for _ in allowed)
            sql = f"SELECT * FROM extraction_jobs WHERE status IN ({placeholders}) ORDER BY created_at DESC"
            with self._connect() as conn:
                rows = conn.execute(sql, allowed).fetchall()
            results = [self._row_to_result(row) for row in rows]

        if fmt == "json":
            payload = [res.model_dump() for res in results]
            return json.dumps(payload, indent=2, ensure_ascii=False), "application/json"

        # Flattened CSV export suitable for ERP/accounting ingestion
        fieldnames = [
            "job_id",
            "filename",
            "doc_type",
            "status",
            "overall_confidence",
            "reconciliation_passed",
            "document_number",
            "document_date",
            "due_date",
            "counterparty_vendor",
            "vendor_npwp",
            "buyer_name",
            "delivery_address",
            "payment_method",
            "currency",
            "line_item_count",
            "line_items_summary",
            "subtotal",
            "tax_amount",
            "total_amount",
            "reviewer_notes",
            "created_at",
            "updated_at",
        ]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()

        for res in results:
            d = res.data
            doc_number = ""
            doc_date = ""
            due_date = ""
            vendor = ""
            npwp = ""
            buyer = ""
            address = ""
            payment = ""
            subtotal: float | str = ""
            tax: float | str = ""
            total: float | str = ""

            if isinstance(d, InvoiceData):
                doc_number = d.invoice_number or ""
                doc_date = d.issue_date or ""
                due_date = d.due_date or ""
                vendor = d.vendor_name or ""
                npwp = d.vendor_npwp or ""
                buyer = d.buyer_name or ""
                subtotal = d.subtotal if d.subtotal is not None else ""
                tax = d.tax_amount if d.tax_amount is not None else ""
                total = d.total_amount if d.total_amount is not None else ""
            elif isinstance(d, PurchaseOrderData):
                doc_number = d.po_number or ""
                doc_date = d.order_date or ""
                vendor = d.supplier_name or ""
                buyer = d.buyer_name or ""
                address = d.delivery_address or ""
                total = d.total_amount if d.total_amount is not None else ""
            elif isinstance(d, ReceiptData):
                doc_number = d.receipt_number or ""
                doc_date = d.transaction_date or ""
                vendor = d.merchant_name or ""
                payment = d.payment_method or ""
                tax = d.tax_amount if d.tax_amount is not None else ""
                total = d.total_amount if d.total_amount is not None else ""

            items_summary = "; ".join(
                f"{item.description} ({item.quantity:g}x{item.unit_price:g}={item.line_total:g})"
                for item in d.line_items
            )
            row_dict = {
                "job_id": res.job_id,
                "filename": res.filename or "",
                "doc_type": res.doc_type,
                "status": res.status,
                "overall_confidence": res.overall_confidence,
                "reconciliation_passed": res.reconciliation_passed,
                "document_number": doc_number,
                "document_date": doc_date,
                "due_date": due_date,
                "counterparty_vendor": vendor,
                "vendor_npwp": npwp,
                "buyer_name": buyer,
                "delivery_address": address,
                "payment_method": payment,
                "currency": d.currency,
                "line_item_count": len(d.line_items),
                "line_items_summary": items_summary,
                "subtotal": subtotal,
                "tax_amount": tax,
                "total_amount": total,
                "reviewer_notes": res.reviewer_notes or "",
                "created_at": res.created_at,
                "updated_at": res.updated_at or "",
            }
            writer.writerow({k: _sanitize_csv_cell(v) for k, v in row_dict.items()})

        return buf.getvalue(), "text/csv; charset=utf-8"
