"""API integration tests using FastAPI TestClient."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from fastapi.testclient import TestClient

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


def test_ui_and_health_endpoints(client: TestClient) -> None:
    ui_resp = client.get("/")
    assert ui_resp.status_code == 200
    assert "DocIntel Extract" in ui_resp.text

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    body = health_resp.json()
    assert body["status"] == "ok"
    assert body["provider"] == "local"
    assert body["review_confidence_threshold"] == 0.80


def test_multipart_pdf_upload_and_hitl_review_flow(client: TestClient) -> None:
    pdf_bytes = (SAMPLES_DIR / "invoice_id_mismatch_01.pdf").read_bytes()
    resp = client.post(
        "/v1/extract",
        files={"file": ("invoice_id_mismatch_01.pdf", pdf_bytes, "application/pdf")},
        data={"doc_type": "auto"},
    )
    assert resp.status_code == 200
    extracted = resp.json()
    assert extracted["doc_type"] == "invoice"
    assert extracted["status"] == "NEEDS_REVIEW"
    assert extracted["reconciliation_passed"] is False
    job_id = extracted["job_id"]

    # Inspect job via GET /v1/jobs/{job_id}
    get_resp = client.get(f"/v1/jobs/{job_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["job_id"] == job_id

    # Human corrects the mismatched total_amount and approves
    rev_resp = client.post(
        f"/v1/jobs/{job_id}/review",
        json={
            "data": {"total_amount": 13875000.0},
            "reviewer_notes": "Verified true sum against PDF line items",
            "approve": True,
        },
    )
    assert rev_resp.status_code == 200
    reviewed = rev_resp.json()
    assert reviewed["status"] == "REVIEWED"
    assert reviewed["reconciliation_passed"] is True
    assert reviewed["overall_confidence"] >= 0.90


def test_json_text_extract_and_batch_and_export(client: TestClient) -> None:
    text_payload = {
        "filename": "quick_po.txt",
        "doc_type": "auto",
        "text": (
            "PURCHASE ORDER\n"
            "PO Number: PO-2026-777\n"
            "Order Date: 2026-05-10\n"
            "Supplier: PT Sumber Jaya\n"
            "Buyer: PT Inovasi AI\n"
            "Delivery Address: Jl. Gatot Subroto No. 12, Jakarta\n"
            "| Description | Qty | Unit Price | Amount |\n"
            "| Server Rack | 2 | Rp 5.000.000 | Rp 10.000.000 |\n"
            "Total Order: Rp 10.000.000\n"
        ),
    }
    resp = client.post("/v1/extract", json=text_payload)
    assert resp.status_code == 200
    po = resp.json()
    assert po["doc_type"] == "purchase_order"
    assert po["status"] == "APPROVED"
    assert po["reconciliation_passed"] is True

    # Batch multipart upload of 2 real PDFs
    inv_bytes = (SAMPLES_DIR / "invoice_en_01.pdf").read_bytes()
    rcpt_bytes = (SAMPLES_DIR / "receipt_en_mismatch_01.pdf").read_bytes()
    batch_resp = client.post(
        "/v1/extract/batch",
        files=[
            ("files", ("invoice_en_01.pdf", inv_bytes, "application/pdf")),
            ("files", ("receipt_en_mismatch_01.pdf", rcpt_bytes, "application/pdf")),
        ],
    )
    assert batch_resp.status_code == 200
    batch = batch_resp.json()
    assert batch["total"] == 2
    assert batch["approved"] == 1
    assert batch["needs_review"] == 1

    # Queue listing & filtering
    all_jobs = client.get("/v1/jobs").json()
    assert len(all_jobs) == 3
    flagged = client.get("/v1/jobs?status=NEEDS_REVIEW").json()
    assert len(flagged) == 1

    # Export JSON (only APPROVED/REVIEWED by default -> 2 records)
    exp_json = client.get("/v1/export?format=json")
    assert exp_json.status_code == 200
    assert len(exp_json.json()) == 2

    # Export CSV
    exp_csv = client.get("/v1/export?format=csv")
    assert exp_csv.status_code == 200
    rows = list(csv.DictReader(io.StringIO(exp_csv.text)))
    assert len(rows) == 2


def test_samples_endpoints(client: TestClient) -> None:
    list_resp = client.get("/v1/samples")
    assert list_resp.status_code == 200
    samples = list_resp.json()["samples"]
    assert len(samples) == 15

    ext_resp = client.post("/v1/samples/invoice_id_01.pdf/extract")
    assert ext_resp.status_code == 200
    assert ext_resp.json()["status"] == "APPROVED"

    missing_resp = client.post("/v1/samples/nonexistent.pdf/extract")
    assert missing_resp.status_code == 404
