"""Unit tests for SQLite HITL ReviewStore and batch JSON/CSV export."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from docintel.pipeline import DocumentIntelligencePipeline
from docintel.schemas import ReviewSubmission

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


def test_store_save_filter_review_and_export(pipeline: DocumentIntelligencePipeline) -> None:
    clean = pipeline.process_file(SAMPLES_DIR / "invoice_id_01.pdf")
    mismatched = pipeline.process_file(SAMPLES_DIR / "invoice_id_mismatch_01.pdf")

    assert clean.status == "APPROVED"
    assert mismatched.status == "NEEDS_REVIEW"

    needs_review_list = pipeline.store.list_jobs(status="NEEDS_REVIEW")
    assert [j.job_id for j in needs_review_list] == [mismatched.job_id]

    counts = pipeline.store.get_status_counts()
    assert counts["APPROVED"] == 1
    assert counts["NEEDS_REVIEW"] == 1
    assert counts["TOTAL"] == 2

    # Submit human review fixing the mismatched grand total (12,500,000 + 1,375,000 = 13,875,000)
    reviewed = pipeline.submit_review(
        mismatched.job_id,
        ReviewSubmission(
            data={"total_amount": 13875000.0},
            reviewer_notes="Fixed OCR digit typo in grand total",
            approve=True,
        ),
    )
    assert reviewed.status == "REVIEWED"
    assert reviewed.reconciliation_passed is True
    assert reviewed.reviewer_notes == "Fixed OCR digit typo in grand total"

    # Export JSON
    json_str, json_mime = pipeline.store.export_records(fmt="json")
    assert json_mime == "application/json"
    exported_json = json.loads(json_str)
    assert len(exported_json) == 2

    # Export CSV
    csv_str, csv_mime = pipeline.store.export_records(fmt="csv")
    assert "text/csv" in csv_mime
    rows = list(csv.DictReader(io.StringIO(csv_str)))
    assert len(rows) == 2
    assert {r["status"] for r in rows} == {"APPROVED", "REVIEWED"}
    assert all(r["reconciliation_passed"] == "True" for r in rows)


def test_csv_export_neutralizes_formula_injection(pipeline: DocumentIntelligencePipeline) -> None:
    text = (
        "COMMERCIAL INVOICE\n"
        "Vendor: =CMD|' /C calc'!A0\n"
        "Bill To: +SUM(1+1)*cmd|' /C calc'!A0\n"
        "Invoice No: INV-2026-SEC01\n"
        "Issue Date: 2026-06-01\n"
        "Due Date: 2026-06-30\n"
        "| Description | Qty | Unit Price | Line Total |\n"
        "| @HYPERLINK(\"http://evil\") | 1 | $100.00 | $100.00 |\n"
        "Subtotal: $100.00\n"
        "VAT (10%): $10.00\n"
        "Grand Total: $110.00\n"
    )
    job = pipeline.process_text(text, filename="=evil_file.txt")
    pipeline.submit_review(
        job.job_id,
        ReviewSubmission(
            data={},
            reviewer_notes="-2+3+cmd|' /C calc'!A0",
            approve=True,
        ),
    )

    csv_str, _ = pipeline.store.export_records(fmt="csv")
    rows = list(csv.DictReader(io.StringIO(csv_str)))
    assert len(rows) == 1
    row = rows[0]
    assert row["filename"].startswith("'=")
    assert row["counterparty_vendor"].startswith("'=")
    assert row["buyer_name"].startswith("'+")
    assert row["line_items_summary"].startswith("'@")
    assert row["reviewer_notes"].startswith("'-")
    # Numeric totals must remain unquoted numbers
    assert row["total_amount"] == "110.0"
