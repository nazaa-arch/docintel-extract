"""Generate 15 realistic multi-layout synthetic PDF documents + ground truth for benchmarking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "data" / "samples"


def _fmt_money(amount: float, currency: str) -> str:
    if currency == "IDR":
        return "Rp " + f"{amount:,.0f}".replace(",", ".")
    if currency == "SGD":
        return f"SGD {amount:,.2f}"
    return f"${amount:,.2f}"


def _draw_bordered_table(
    page: fitz.Page,
    x: float,
    y: float,
    col_widths: list[float],
    headers: list[str],
    rows: list[list[str]],
    row_height: float = 22.0,
) -> float:
    """Draw a ruled vector table so pdfplumber and PyMuPDF both extract clean cells."""
    total_w = sum(col_widths)
    all_rows = [headers, *rows]
    cur_y = y

    for r_idx, row in enumerate(all_rows):
        rect = fitz.Rect(x, cur_y, x + total_w, cur_y + row_height)
        if r_idx == 0:
            page.draw_rect(rect, color=(0.2, 0.25, 0.35), fill=(0.91, 0.94, 0.98), width=0.8)
        else:
            page.draw_rect(rect, color=(0.65, 0.7, 0.78), width=0.6)

        cur_x = x
        for c_idx, (cell, width) in enumerate(zip(row, col_widths, strict=True)):
            if c_idx > 0:
                page.draw_line(
                    fitz.Point(cur_x, cur_y),
                    fitz.Point(cur_x, cur_y + row_height),
                    color=(0.65, 0.7, 0.78),
                    width=0.6,
                )
            fontname = "helv" if r_idx > 0 else "hebo"
            fontsize = 9.0 if r_idx > 0 else 9.5
            page.insert_text(
                fitz.Point(cur_x + 5, cur_y + 14.5),
                cell,
                fontname=fontname,
                fontsize=fontsize,
                color=(0.1, 0.12, 0.16),
            )
            cur_x += width
        cur_y += row_height

    return cur_y


def _draw_borderless_table(
    page: fitz.Page,
    x: float,
    y: float,
    col_offsets: list[float],
    headers: list[str],
    rows: list[list[str]],
    row_height: float = 20.0,
) -> float:
    """Draw a modern borderless table (horizontal divider only) to exercise layout reconstruction."""
    cur_y = y
    for offset, head in zip(col_offsets, headers, strict=True):
        page.insert_text(fitz.Point(x + offset, cur_y), head, fontname="hebo", fontsize=9.5, color=(0.15, 0.2, 0.3))
    cur_y += 6
    page.draw_line(fitz.Point(x, cur_y), fitz.Point(555, cur_y), color=(0.3, 0.35, 0.45), width=1.0)
    cur_y += 16

    for row in rows:
        for offset, cell in zip(col_offsets, row, strict=True):
            page.insert_text(fitz.Point(x + offset, cur_y), cell, fontname="helv", fontsize=9.0, color=(0.1, 0.12, 0.15))
        cur_y += row_height

    page.draw_line(fitz.Point(x, cur_y - 6), fitz.Point(555, cur_y - 6), color=(0.7, 0.75, 0.8), width=0.6)
    return cur_y + 4


def _render_invoice_pdf(spec: dict[str, Any], out_path: Path) -> None:
    with fitz.open() as doc:
        page = doc.new_page(width=595, height=842)
        gt = spec["ground_truth"]
        lang = spec["lang"]
        bordered = spec["layout"] == "bordered"
        cur = gt["currency"]

        title = "FAKTUR PENJUALAN / INVOICE" if lang == "id" else "COMMERCIAL INVOICE"
        page.insert_text(fitz.Point(40, 48), title, fontname="hebo", fontsize=16, color=(0.08, 0.18, 0.36))
        page.draw_line(fitz.Point(40, 56), fitz.Point(555, 56), color=(0.12, 0.45, 0.75), width=1.6)

        y = 80.0
        if lang == "id":
            header_lines = [
                f"Penjual: {gt['vendor_name']}",
                f"NPWP: {gt['vendor_npwp']}" if gt.get("vendor_npwp") else "",
                f"Tagihan Kepada: {gt['buyer_name']}",
                f"No. Faktur: {gt['invoice_number']}" if gt.get("invoice_number") else "",
                f"Tanggal Faktur: {spec['display_issue_date']}",
                f"Jatuh Tempo: {spec['display_due_date']}",
            ]
        else:
            header_lines = [
                f"Vendor: {gt['vendor_name']}",
                f"Bill To: {gt['buyer_name']}",
                f"Invoice No: {gt['invoice_number']}" if gt.get("invoice_number") else "",
                f"Issue Date: {spec['display_issue_date']}",
                f"Due Date: {spec['display_due_date']}",
            ]

        for line in header_lines:
            if not line:
                continue
            page.insert_text(fitz.Point(40, y), line, fontname="helv", fontsize=10.0, color=(0.12, 0.14, 0.18))
            y += 17.0

        y += 12.0
        headers = ["Deskripsi", "Jumlah", "Harga Satuan", "Total"] if lang == "id" else ["Description", "Qty", "Unit Price", "Line Total"]
        rows = [
            [
                it["description"],
                f"{it['quantity']:g}",
                _fmt_money(it["unit_price"], cur),
                _fmt_money(it["line_total"], cur),
            ]
            for it in gt["line_items"]
        ]

        if bordered:
            y = _draw_bordered_table(page, 40, y, [245, 55, 105, 110], headers, rows)
        else:
            y = _draw_borderless_table(page, 40, y, [0, 260, 325, 425], headers, rows)

        y += 18.0
        tax_label = f"PPN {spec.get('tax_pct', 11)}%" if lang == "id" else f"VAT / Tax ({spec.get('tax_pct', 10)}%)"
        total_label = "Total Tagihan" if lang == "id" else "Grand Total"

        summary_rows = [
            ("Subtotal", _fmt_money(gt["subtotal"], cur)),
            (tax_label, _fmt_money(gt["tax_amount"], cur)),
            (total_label, _fmt_money(gt["total_amount"], cur)),
        ]
        for idx, (label, val) in enumerate(summary_rows):
            font = "hebo" if idx == 2 else "helv"
            size = 11.0 if idx == 2 else 10.0
            page.insert_text(fitz.Point(320, y), f"{label}:", fontname=font, fontsize=size, color=(0.1, 0.12, 0.16))
            page.insert_text(fitz.Point(435, y), val, fontname=font, fontsize=size, color=(0.1, 0.12, 0.16))
            y += 18.0

        doc.save(out_path)


def _render_po_pdf(spec: dict[str, Any], out_path: Path) -> None:
    with fitz.open() as doc:
        page = doc.new_page(width=595, height=842)
        gt = spec["ground_truth"]
        lang = spec["lang"]
        bordered = spec["layout"] == "bordered"
        cur = gt["currency"]

        title = "PESANAN PEMBELIAN / PURCHASE ORDER" if lang == "id" else "PURCHASE ORDER"
        page.insert_text(fitz.Point(40, 48), title, fontname="hebo", fontsize=16, color=(0.12, 0.25, 0.42))
        page.draw_line(fitz.Point(40, 56), fitz.Point(555, 56), color=(0.2, 0.4, 0.65), width=1.6)

        y = 80.0
        if lang == "id":
            header_lines = [
                f"No. PO: {gt['po_number']}",
                f"Tanggal Pesanan: {spec['display_order_date']}",
                f"Pemasok: {gt['supplier_name']}",
                f"Pembeli: {gt['buyer_name']}",
                f"Alamat Pengiriman: {gt['delivery_address']}",
            ]
        else:
            header_lines = [
                f"PO Number: {gt['po_number']}",
                f"Order Date: {spec['display_order_date']}",
                f"Supplier: {gt['supplier_name']}",
                f"Buyer: {gt['buyer_name']}",
                f"Delivery Address: {gt['delivery_address']}",
            ]

        for line in header_lines:
            page.insert_text(fitz.Point(40, y), line, fontname="helv", fontsize=10.0, color=(0.12, 0.14, 0.18))
            y += 17.0

        y += 12.0
        headers = ["Barang / Deskripsi", "Qty", "Harga Satuan", "Jumlah Harga"] if lang == "id" else ["Item Description", "Qty", "Unit Price", "Amount"]
        rows = [
            [
                it["description"],
                f"{it['quantity']:g}",
                _fmt_money(it["unit_price"], cur),
                _fmt_money(it["line_total"], cur),
            ]
            for it in gt["line_items"]
        ]

        if bordered:
            y = _draw_bordered_table(page, 40, y, [245, 55, 105, 110], headers, rows)
        else:
            y = _draw_borderless_table(page, 40, y, [0, 260, 325, 425], headers, rows)

        y += 20.0
        total_label = "Total Pesanan" if lang == "id" else "Total Order"
        page.insert_text(fitz.Point(320, y), f"{total_label}:", fontname="hebo", fontsize=11.0, color=(0.1, 0.12, 0.16))
        page.insert_text(fitz.Point(435, y), _fmt_money(gt["total_amount"], cur), fontname="hebo", fontsize=11.0, color=(0.1, 0.12, 0.16))

        doc.save(out_path)


def _render_receipt_pdf(spec: dict[str, Any], out_path: Path) -> None:
    with fitz.open() as doc:
        # Narrow thermal-receipt PDF dimensions
        page = doc.new_page(width=360, height=560)
        gt = spec["ground_truth"]
        lang = spec["lang"]
        cur = gt["currency"]

        title = "STRUK PEMBAYARAN" if lang == "id" else "SALES RECEIPT"
        page.insert_text(fitz.Point(24, 34), title, fontname="hebo", fontsize=13, color=(0.1, 0.12, 0.16))
        page.draw_line(fitz.Point(24, 42), fitz.Point(336, 42), color=(0.3, 0.3, 0.3), width=1.0)

        y = 62.0
        if lang == "id":
            meta_lines = [
                f"Merchant: {gt['merchant_name']}",
                f"No. Struk: {gt['receipt_number']}",
                f"Tanggal Transaksi: {spec['display_tx_date']}",
                f"Metode Pembayaran: {gt['payment_method']}",
            ]
        else:
            meta_lines = [
                f"Merchant: {gt['merchant_name']}",
                f"Receipt No: {gt['receipt_number']}",
                f"Transaction Date: {spec['display_tx_date']}",
                f"Payment Method: {gt['payment_method']}",
            ]

        for line in meta_lines:
            page.insert_text(fitz.Point(24, y), line, fontname="helv", fontsize=9.5, color=(0.12, 0.14, 0.18))
            y += 15.0

        y += 6.0
        page.draw_line(fitz.Point(24, y), fitz.Point(336, y), color=(0.6, 0.6, 0.6), width=0.8)
        y += 16.0

        headers = ["Item", "Qty", "Harga", "Total"] if lang == "id" else ["Item", "Qty", "Price", "Total"]
        offsets = [0, 140, 180, 248]
        for off, h in zip(offsets, headers, strict=True):
            page.insert_text(fitz.Point(24 + off, y), h, fontname="hebo", fontsize=8.5, color=(0.15, 0.18, 0.22))
        y += 15.0

        for it in gt["line_items"]:
            vals = [
                it["description"],
                f"{it['quantity']:g}",
                _fmt_money(it["unit_price"], cur),
                _fmt_money(it["line_total"], cur),
            ]
            for off, val in zip(offsets, vals, strict=True):
                page.insert_text(fitz.Point(24 + off, y), val, fontname="helv", fontsize=8.5, color=(0.1, 0.12, 0.15))
            y += 15.0

        y += 4.0
        page.draw_line(fitz.Point(24, y), fitz.Point(336, y), color=(0.6, 0.6, 0.6), width=0.8)
        y += 18.0

        tax_label = "PB1 (10%)" if lang == "id" else "Sales Tax (8%)"
        total_label = "Total Bayar" if lang == "id" else "Grand Total"
        page.insert_text(fitz.Point(145, y), f"{tax_label}:", fontname="helv", fontsize=9.0)
        page.insert_text(fitz.Point(255, y), _fmt_money(gt["tax_amount"], cur), fontname="helv", fontsize=9.0)
        y += 17.0
        page.insert_text(fitz.Point(145, y), f"{total_label}:", fontname="hebo", fontsize=10.5)
        page.insert_text(fitz.Point(255, y), _fmt_money(gt["total_amount"], cur), fontname="hebo", fontsize=10.5)
        y += 28.0

        footer = "Terima kasih telah berbelanja!" if lang == "id" else "Thank you for your purchase!"
        page.insert_text(fitz.Point(24, y), footer, fontname="helv", fontsize=8.5, color=(0.35, 0.4, 0.48))

        doc.save(out_path)


SAMPLE_SPECS: list[dict[str, Any]] = [
    # --- 11 CLEAN DOCUMENTS (expected APPROVED) ---
    {
        "filename": "invoice_id_01.pdf",
        "doc_type": "invoice",
        "lang": "id",
        "layout": "bordered",
        "tax_pct": 11,
        "display_issue_date": "15 Maret 2026",
        "display_due_date": "14 April 2026",
        "expected_status": "APPROVED",
        "expected_reconciliation_passed": True,
        "degradation_type": "none",
        "ground_truth": {
            "invoice_number": "INV-2026-0301",
            "issue_date": "2026-03-15",
            "due_date": "2026-04-14",
            "vendor_name": "PT Nusantara Cloud Teknika",
            "vendor_npwp": "01.234.567.8-012.000",
            "buyer_name": "PT Bank Karya Digital Tbk",
            "currency": "IDR",
            "line_items": [
                {"description": "GPU Inference Cluster A100", "quantity": 2.0, "unit_price": 4500000.0, "line_total": 9000000.0},
                {"description": "Managed Vector Database", "quantity": 1.0, "unit_price": 3500000.0, "line_total": 3500000.0},
            ],
            "subtotal": 12500000.0,
            "tax_amount": 1375000.0,
            "total_amount": 13875000.0,
        },
    },
    {
        "filename": "invoice_id_02.pdf",
        "doc_type": "invoice",
        "lang": "id",
        "layout": "borderless",
        "tax_pct": 11,
        "display_issue_date": "02/04/2026",
        "display_due_date": "02/05/2026",
        "expected_status": "APPROVED",
        "expected_reconciliation_passed": True,
        "degradation_type": "none",
        "ground_truth": {
            "invoice_number": "FKT-2026-0412",
            "issue_date": "2026-04-02",
            "due_date": "2026-05-02",
            "vendor_name": "PT Sinergi Data Nusantara",
            "vendor_npwp": "02.881.450.3-041.000",
            "buyer_name": "PT Mandiri Logistik Prima",
            "currency": "IDR",
            "line_items": [
                {"description": "Lisensi OCR Enterprise", "quantity": 3.0, "unit_price": 2000000.0, "line_total": 6000000.0},
                {"description": "Integrasi API ERP", "quantity": 1.0, "unit_price": 4000000.0, "line_total": 4000000.0},
            ],
            "subtotal": 10000000.0,
            "tax_amount": 1100000.0,
            "total_amount": 11100000.0,
        },
    },
    {
        "filename": "invoice_id_03.pdf",
        "doc_type": "invoice",
        "lang": "id",
        "layout": "bordered",
        "tax_pct": 11,
        "display_issue_date": "20 Mei 2026",
        "display_due_date": "19 Juni 2026",
        "expected_status": "APPROVED",
        "expected_reconciliation_passed": True,
        "degradation_type": "none",
        "ground_truth": {
            "invoice_number": "INV-2026-0520",
            "issue_date": "2026-05-20",
            "due_date": "2026-06-19",
            "vendor_name": "CV Mitra Karya Sejahtera",
            "vendor_npwp": "03.451.992.1-073.000",
            "buyer_name": "PT Retail Jaya Abadi",
            "currency": "IDR",
            "line_items": [
                {"description": "Barcode Scanner Wireless", "quantity": 5.0, "unit_price": 650000.0, "line_total": 3250000.0},
                {"description": "Thermal Printer 80mm", "quantity": 2.0, "unit_price": 1200000.0, "line_total": 2400000.0},
                {"description": "Kertas Struk Thermal Box", "quantity": 10.0, "unit_price": 85000.0, "line_total": 850000.0},
            ],
            "subtotal": 6500000.0,
            "tax_amount": 715000.0,
            "total_amount": 7215000.0,
        },
    },
    {
        "filename": "invoice_en_01.pdf",
        "doc_type": "invoice",
        "lang": "en",
        "layout": "bordered",
        "tax_pct": 8,
        "display_issue_date": "March 10, 2026",
        "display_due_date": "April 09, 2026",
        "expected_status": "APPROVED",
        "expected_reconciliation_passed": True,
        "degradation_type": "none",
        "ground_truth": {
            "invoice_number": "INV-US-8841",
            "issue_date": "2026-03-10",
            "due_date": "2026-04-09",
            "vendor_name": "Vertex AI Infrastructure Inc.",
            "vendor_npwp": None,
            "buyer_name": "Pacific FinTech Holdings",
            "currency": "USD",
            "line_items": [
                {"description": "Dedicated H100 Node Hours", "quantity": 40.0, "unit_price": 32.5, "line_total": 1300.0},
                {"description": "Enterprise Telemetry Addon", "quantity": 1.0, "unit_price": 450.0, "line_total": 450.0},
            ],
            "subtotal": 1750.0,
            "tax_amount": 140.0,
            "total_amount": 1890.0,
        },
    },
    {
        "filename": "invoice_en_02.pdf",
        "doc_type": "invoice",
        "lang": "en",
        "layout": "borderless",
        "tax_pct": 10,
        "display_issue_date": "2026-04-18",
        "display_due_date": "2026-05-18",
        "expected_status": "APPROVED",
        "expected_reconciliation_passed": True,
        "degradation_type": "none",
        "ground_truth": {
            "invoice_number": "INV-EN-9012",
            "issue_date": "2026-04-18",
            "due_date": "2026-05-18",
            "vendor_name": "Meridian Applied Systems LLC",
            "vendor_npwp": None,
            "buyer_name": "Atlas Commerce Group",
            "currency": "USD",
            "line_items": [
                {"description": "RAG Pipeline Architecture Audit", "quantity": 2.0, "unit_price": 1200.0, "line_total": 2400.0},
                {"description": "Eval Harness Setup", "quantity": 1.0, "unit_price": 800.0, "line_total": 800.0},
            ],
            "subtotal": 3200.0,
            "tax_amount": 320.0,
            "total_amount": 3520.0,
        },
    },
    {
        "filename": "po_id_01.pdf",
        "doc_type": "purchase_order",
        "lang": "id",
        "layout": "bordered",
        "display_order_date": "12 Maret 2026",
        "expected_status": "APPROVED",
        "expected_reconciliation_passed": True,
        "degradation_type": "none",
        "ground_truth": {
            "po_number": "PO-2026-0118",
            "order_date": "2026-03-12",
            "supplier_name": "PT Distribusi Komputer Sentosa",
            "buyer_name": "PT Teknologi Cipta Mandiri",
            "delivery_address": "Jl. Jenderal Sudirman Kav. 52, Jakarta Selatan",
            "currency": "IDR",
            "line_items": [
                {"description": "Workstation ThinkStation P3", "quantity": 3.0, "unit_price": 18500000.0, "line_total": 55500000.0},
                {"description": "Monitor 27 Inch 4K IPS", "quantity": 6.0, "unit_price": 4500000.0, "line_total": 27000000.0},
            ],
            "total_amount": 82500000.0,
        },
    },
    {
        "filename": "po_id_02.pdf",
        "doc_type": "purchase_order",
        "lang": "id",
        "layout": "borderless",
        "display_order_date": "05/04/2026",
        "expected_status": "APPROVED",
        "expected_reconciliation_passed": True,
        "degradation_type": "none",
        "ground_truth": {
            "po_number": "PO-2026-0244",
            "order_date": "2026-04-05",
            "supplier_name": "PT Cahaya Elektrik Industri",
            "buyer_name": "PT Manufaktur Timur Jaya",
            "delivery_address": "Kawasan Industri Rungkut Blok C4, Surabaya",
            "currency": "IDR",
            "line_items": [
                {"description": "UPS Rackmount 3000VA", "quantity": 4.0, "unit_price": 6250000.0, "line_total": 25000000.0},
                {"description": "Kabel Patch Cat6A 5m", "quantity": 20.0, "unit_price": 125000.0, "line_total": 2500000.0},
            ],
            "total_amount": 27500000.0,
        },
    },
    {
        "filename": "po_en_01.pdf",
        "doc_type": "purchase_order",
        "lang": "en",
        "layout": "bordered",
        "display_order_date": "April 01, 2026",
        "expected_status": "APPROVED",
        "expected_reconciliation_passed": True,
        "degradation_type": "none",
        "ground_truth": {
            "po_number": "PO-EN-7701",
            "order_date": "2026-04-01",
            "supplier_name": "CoreLink Hardware Supply Ltd",
            "buyer_name": "Stratos Cloud Operations",
            "delivery_address": "88 Marina Boulevard Tower 2, Singapore 018981",
            "currency": "USD",
            "line_items": [
                {"description": "NVMe Enterprise SSD 7.68TB", "quantity": 4.0, "unit_price": 820.0, "line_total": 3280.0},
                {"description": "100GbE QSFP28 Transceiver", "quantity": 8.0, "unit_price": 145.0, "line_total": 1160.0},
            ],
            "total_amount": 4440.0,
        },
    },
    {
        "filename": "po_en_02.pdf",
        "doc_type": "purchase_order",
        "lang": "en",
        "layout": "borderless",
        "display_order_date": "2026-05-03",
        "expected_status": "APPROVED",
        "expected_reconciliation_passed": True,
        "degradation_type": "none",
        "ground_truth": {
            "po_number": "PO-SG-3092",
            "order_date": "2026-05-03",
            "supplier_name": "Apex Office Ergonomics Pte Ltd",
            "buyer_name": "Horizon Labs Asia",
            "delivery_address": "12 One-North Gateway #04-11, Singapore",
            "currency": "SGD",
            "line_items": [
                {"description": "Standing Desk Motorized", "quantity": 5.0, "unit_price": 680.0, "line_total": 3400.0},
                {"description": "Ergonomic Mesh Chair", "quantity": 5.0, "unit_price": 420.0, "line_total": 2100.0},
            ],
            "total_amount": 5500.0,
        },
    },
    {
        "filename": "receipt_id_01.pdf",
        "doc_type": "receipt",
        "lang": "id",
        "layout": "thermal",
        "display_tx_date": "18 Maret 2026",
        "expected_status": "APPROVED",
        "expected_reconciliation_passed": True,
        "degradation_type": "none",
        "ground_truth": {
            "merchant_name": "Kopi Kenangan Senopati",
            "transaction_date": "2026-03-18",
            "receipt_number": "STR-2026-8812",
            "payment_method": "QRIS BCA",
            "currency": "IDR",
            "line_items": [
                {"description": "Kopi Susu Aren", "quantity": 2.0, "unit_price": 25000.0, "line_total": 50000.0},
                {"description": "Butter Croissant", "quantity": 2.0, "unit_price": 30000.0, "line_total": 60000.0},
            ],
            "tax_amount": 11000.0,
            "total_amount": 121000.0,
        },
    },
    {
        "filename": "receipt_en_01.pdf",
        "doc_type": "receipt",
        "lang": "en",
        "layout": "thermal",
        "display_tx_date": "2026-04-22",
        "expected_status": "APPROVED",
        "expected_reconciliation_passed": True,
        "degradation_type": "none",
        "ground_truth": {
            "merchant_name": "Metro Tech Supply Store",
            "transaction_date": "2026-04-22",
            "receipt_number": "RCPT-44910",
            "payment_method": "Visa Credit Card",
            "currency": "USD",
            "line_items": [
                {"description": "USB-C Cable 2m", "quantity": 2.0, "unit_price": 15.0, "line_total": 30.0},
                {"description": "GaN Charger 65W", "quantity": 1.0, "unit_price": 45.0, "line_total": 45.0},
            ],
            "tax_amount": 6.0,
            "total_amount": 81.0,
        },
    },
    # --- 4 INTENTIONALLY DEGRADED / MISMATCHED DOCUMENTS (expected NEEDS_REVIEW) ---
    {
        "filename": "invoice_id_mismatch_01.pdf",
        "doc_type": "invoice",
        "lang": "id",
        "layout": "bordered",
        "tax_pct": 11,
        "display_issue_date": "25 Maret 2026",
        "display_due_date": "24 April 2026",
        "expected_status": "NEEDS_REVIEW",
        "expected_reconciliation_passed": False,
        "degradation_type": "total_mismatch",
        "ground_truth": {
            "invoice_number": "INV-2026-9901",
            "issue_date": "2026-03-25",
            "due_date": "2026-04-24",
            "vendor_name": "PT Solusi Infra Digital",
            "vendor_npwp": "04.512.880.9-021.000",
            "buyer_name": "PT Finansi Global Indonesia",
            "currency": "IDR",
            "line_items": [
                {"description": "Server Colocation Rack", "quantity": 1.0, "unit_price": 8000000.0, "line_total": 8000000.0},
                {"description": "Dedicated Fiber 1Gbps", "quantity": 1.0, "unit_price": 4500000.0, "line_total": 4500000.0},
            ],
            "subtotal": 12500000.0,
            "tax_amount": 1375000.0,
            # Deliberate clerical/OCR discrepancy (+375,000 Rp vs true 13,875,000)
            "total_amount": 14250000.0,
        },
    },
    {
        "filename": "invoice_en_missing_01.pdf",
        "doc_type": "invoice",
        "lang": "en",
        "layout": "borderless",
        "tax_pct": 10,
        "display_issue_date": "TBD",
        "display_due_date": "2026-05-30",
        "expected_status": "NEEDS_REVIEW",
        "expected_reconciliation_passed": True,
        "degradation_type": "missing_fields",
        "ground_truth": {
            # Deliberately missing invoice_number and unparseable issue_date ("TBD")
            "invoice_number": None,
            "issue_date": "TBD",
            "due_date": "2026-05-30",
            "vendor_name": "CloudBridge Advisory Partners",
            "vendor_npwp": None,
            "buyer_name": "Northstar Logistics Corp",
            "currency": "USD",
            "line_items": [
                {"description": "Security Compliance Review", "quantity": 2.0, "unit_price": 900.0, "line_total": 1800.0},
            ],
            "subtotal": 1800.0,
            "tax_amount": 180.0,
            "total_amount": 1980.0,
        },
    },
    {
        "filename": "po_id_mismatch_01.pdf",
        "doc_type": "purchase_order",
        "lang": "id",
        "layout": "bordered",
        "display_order_date": "11 April 2026",
        "expected_status": "NEEDS_REVIEW",
        "expected_reconciliation_passed": False,
        "degradation_type": "line_item_mismatch",
        "ground_truth": {
            "po_number": "PO-2026-9955",
            "order_date": "2026-04-11",
            "supplier_name": "PT Anugerah Peralatan Kantor",
            "buyer_name": "PT Karya Kreatif Nusantara",
            "delivery_address": "Jl. HR Rasuna Said Kav. 10, Jakarta Selatan",
            "currency": "IDR",
            "line_items": [
                # Deliberate line-item math error: 5 * 450,000 = 2,250,000 != 2,500,000
                {"description": "Kursi Kerja Ergonomis", "quantity": 5.0, "unit_price": 450000.0, "line_total": 2500000.0},
                {"description": "Meja Meeting Kayu", "quantity": 1.0, "unit_price": 3500000.0, "line_total": 3500000.0},
            ],
            "total_amount": 6000000.0,
        },
    },
    {
        "filename": "receipt_en_mismatch_01.pdf",
        "doc_type": "receipt",
        "lang": "en",
        "layout": "thermal",
        "display_tx_date": "2026-05-09",
        "expected_status": "NEEDS_REVIEW",
        "expected_reconciliation_passed": False,
        "degradation_type": "receipt_total_mismatch",
        "ground_truth": {
            "merchant_name": "Downtown Artisan Roasters",
            "transaction_date": "2026-05-09",
            "receipt_number": "RCPT-90811",
            "payment_method": "Mastercard Debit",
            "currency": "USD",
            "line_items": [
                {"description": "Cold Brew Bottle", "quantity": 3.0, "unit_price": 6.5, "line_total": 19.5},
                {"description": "Sourdough Sandwich", "quantity": 2.0, "unit_price": 11.5, "line_total": 23.0},
            ],
            "tax_amount": 3.4,
            # Deliberate total discrepancy: 42.50 + 3.40 = 45.90 != 55.90
            "total_amount": 55.9,
        },
    },
]


def generate_all_samples(output_dir: Path = OUTPUT_DIR) -> Path:
    """Render all 15 sample PDFs and write ground_truth.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_records: list[dict[str, Any]] = []

    for spec in SAMPLE_SPECS:
        pdf_path = output_dir / spec["filename"]
        doc_type = spec["doc_type"]
        if doc_type == "invoice":
            _render_invoice_pdf(spec, pdf_path)
        elif doc_type == "purchase_order":
            _render_po_pdf(spec, pdf_path)
        else:
            _render_receipt_pdf(spec, pdf_path)

        gt_records.append(
            {
                "filename": spec["filename"],
                "doc_type": doc_type,
                "lang": spec["lang"],
                "layout": spec["layout"],
                "expected_status": spec["expected_status"],
                "expected_reconciliation_passed": spec["expected_reconciliation_passed"],
                "degradation_type": spec["degradation_type"],
                "ground_truth": spec["ground_truth"],
            }
        )

    gt_path = output_dir / "ground_truth.json"
    gt_path.write_text(json.dumps(gt_records, indent=2, ensure_ascii=False), encoding="utf-8")
    return gt_path


if __name__ == "__main__":
    path = generate_all_samples()
    print(f"Generated {len(SAMPLE_SPECS)} real PDFs and ground truth at: {path}")
