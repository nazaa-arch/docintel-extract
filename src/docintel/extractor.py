"""Hybrid deterministic layout/table extractor + pluggable Vision/LLM adapters."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from docintel.config import Settings
from docintel.ingestion import IngestedDocument
from docintel.schemas import (
    DocType,
    DocTypeRequest,
    DocumentData,
    InvoiceData,
    LineItem,
    PurchaseOrderData,
    ReceiptData,
    schema_for_doc_type,
)

MONTH_MAP: dict[str, int] = {
    # English
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
    # Indonesian
    "januari": 1,
    "februari": 2,
    "pebruari": 2,
    "maret": 3,
    "mei": 5,
    "juni": 6,
    "juli": 7,
    "agu": 8,
    "agt": 8,
    "agustus": 8,
    "okt": 10,
    "oktober": 10,
    "nop": 11,
    "nopember": 11,
    "des": 12,
    "desember": 12,
}

DOC_TITLE_NOISE = {
    "invoice",
    "tax invoice",
    "commercial invoice",
    "faktur",
    "faktur pajak",
    "faktur penjualan",
    "tagihan",
    "purchase order",
    "pesanan pembelian",
    "order pembelian",
    "receipt",
    "sales receipt",
    "payment receipt",
    "struk",
    "struk pembayaran",
    "struk belanja",
    "kwitansi",
    "nota",
    "nota penjualan",
}

SUMMARY_ROW_KEYWORDS = (
    "subtotal",
    "sub total",
    "sub-total",
    "dpp",
    "grand total",
    "total amount",
    "total due",
    "amount due",
    "total tagihan",
    "total bayar",
    "total pembayaran",
    "total order",
    "total po",
    "ppn",
    "vat",
    "sales tax",
    "pajak",
    "pb1",
    "tax",
    "discount",
    "diskon",
    "cash",
    "tunai",
    "change",
    "kembali",
    "kembalian",
    "bayar",
    "paid",
)

TABLE_HEADER_KEYWORDS = (
    "description",
    "deskripsi",
    "uraian",
    "item",
    "barang",
    "produk",
    "menu",
    "service",
    "qty",
    "quantity",
    "jumlah",
    "unit price",
    "harga satuan",
    "harga",
    "price",
    "rate",
    "line total",
    "amount",
    "total",
)

NPWP_PATTERN = re.compile(r"\b\d{2}\.\d{3}\.\d{3}\.\d-\d{3}\.\d{3}\b")
NPWP_16_PATTERN = re.compile(r"\b\d{4}\.\d{4}\.\d{4}\.\d{4}\b|\b\d{16}\b")
MONEY_TOKEN_PATTERN = re.compile(
    r"(?:Rp\.?|IDR|USD|EUR|SGD|GBP|MYR|\$|€|£)?\s*-?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?(?:,-|/-)?|"
    r"(?:Rp\.?|IDR|USD|EUR|SGD|GBP|MYR|\$|€|£)\s*-?\d+(?:[.,]\d{1,2})?(?:,-|/-)?|"
    r"-?\d+(?:[.,]\d{1,2})?"
)


@dataclass
class RawExtractionOutput:
    """Internal result from an extraction backend before validation & scoring."""

    doc_type: DocType
    data: DocumentData
    field_sources: dict[str, str] = field(default_factory=dict)
    raw_fields: dict[str, Any] = field(default_factory=dict)
    provider: str = "local"


def detect_currency(text: str) -> str:
    """Detect dominant currency code from document text."""
    if re.search(r"\bIDR\b|Rp\.?\s*\d|\bRupiah\b", text, re.IGNORECASE):
        return "IDR"
    if re.search(r"\bSGD\b|S\$\s*\d", text, re.IGNORECASE):
        return "SGD"
    if re.search(r"\bEUR\b|€\s*\d", text, re.IGNORECASE):
        return "EUR"
    if re.search(r"\bGBP\b|£\s*\d", text, re.IGNORECASE):
        return "GBP"
    if re.search(r"\bMYR\b|\bRM\s*\d", text, re.IGNORECASE):
        return "MYR"
    if re.search(r"\bUSD\b|\$\s*\d", text, re.IGNORECASE):
        return "USD"
    return "IDR"


def parse_money(raw: str | float | int | None) -> float | None:
    """Parse Indonesian (Rp 1.250.000,00) and English ($1,250.00) currency strings into float."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)

    cleaned = str(raw).strip()
    if not cleaned or "%" in cleaned:
        return None

    # Strip currency markers and Indonesian trailing ",-"
    cleaned = re.sub(r"(?i)\b(?:IDR|USD|EUR|SGD|GBP|MYR|RM|Rp\.?)\b", "", cleaned)
    cleaned = re.sub(r"[Rp$€£\s]", "", cleaned)
    cleaned = re.sub(r"(?:,-|/-)$", "", cleaned)
    cleaned = cleaned.strip().rstrip(".")

    if not cleaned or not re.search(r"\d", cleaned):
        return None

    negative = cleaned.startswith("-") or (cleaned.startswith("(") and cleaned.endswith(")"))
    cleaned = cleaned.strip("()-+")
    if not re.fullmatch(r"[\d.,]+", cleaned):
        return None

    dot_count = cleaned.count(".")
    comma_count = cleaned.count(",")

    if dot_count > 0 and comma_count > 0:
        # Whichever separator appears last is the decimal separator
        if cleaned.rfind(",") > cleaned.rfind("."):
            normalized = cleaned.replace(".", "").replace(",", ".")
        else:
            normalized = cleaned.replace(",", "")
    elif comma_count > 0:
        if comma_count > 1:
            normalized = cleaned.replace(",", "")
        else:
            left, right = cleaned.split(",", 1)
            normalized = f"{left}{right}" if len(right) == 3 else f"{left}.{right}"
    elif dot_count > 0:
        if dot_count > 1:
            normalized = cleaned.replace(".", "")
        else:
            left, right = cleaned.split(".", 1)
            normalized = f"{left}{right}" if len(right) == 3 else f"{left}.{right}"
    else:
        normalized = cleaned

    try:
        value = float(normalized)
        return -value if negative else value
    except ValueError:
        return None


def parse_date(raw: str | None) -> str | None:
    """Normalize ISO, slash, English, and Indonesian dates into YYYY-MM-DD."""
    if not raw:
        return None
    text = " ".join(str(raw).strip().split())

    # 1. ISO YYYY-MM-DD or YYYY/MM/DD
    m = re.search(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", text)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    # 2. DD Month YYYY (English or Indonesian)
    m = re.search(r"\b(\d{1,2})\s+([A-Za-z]+)\.?,?\s+(\d{4})\b", text)
    if m:
        day = int(m.group(1))
        month = MONTH_MAP.get(m.group(2).lower())
        year = int(m.group(3))
        if month:
            try:
                return datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                return None

    # 3. Month DD, YYYY
    m = re.search(r"\b([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})\b", text)
    if m:
        month = MONTH_MAP.get(m.group(1).lower())
        day = int(m.group(2))
        year = int(m.group(3))
        if month:
            try:
                return datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                return None

    # 4. DD/MM/YYYY or DD-MM-YYYY (also handles US MM/DD/YYYY when second part > 12)
    m = re.search(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b", text)
    if m:
        first, second, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if first > 12 and second <= 12:
            day, month = first, second
        elif second > 12 and first <= 12:
            month, day = first, second
        else:
            day, month = first, second
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    return None


def classify_document(doc: IngestedDocument) -> DocType:
    """Classify an ingested document into invoice, purchase_order, or receipt."""
    text_lower = doc.raw_text.lower()
    header_lower = "\n".join(doc.lines[:7]).lower()

    signals: dict[DocType, list[tuple[str, float]]] = {
        "invoice": [
            (r"\binvoice\b", 4.0),
            (r"\bfaktur\b", 4.5),
            (r"\btagihan\b", 3.0),
            (r"\binv[-/]\d", 3.5),
            (r"\bfkt[-/]\d", 3.5),
            (r"\bdue date\b|\bjatuh tempo\b", 3.0),
            (r"\bnpwp\b", 2.0),
            (r"\bbill to\b|\btagihan kepada\b|\bkepada yth\b", 2.0),
            (r"\bsubtotal\b|\bsub total\b", 1.5),
            (r"\bppn\b|\bvat\b", 1.5),
        ],
        "purchase_order": [
            (r"\bpurchase order\b", 6.0),
            (r"\bpesanan pembelian\b|\border pembelian\b", 6.0),
            (r"\bpo number\b|\bpo no\.?\b|\bno\.?\s*po\b", 5.0),
            (r"\bpo[-/]\d", 4.0),
            (r"\bdelivery address\b|\balamat pengiriman\b|\bship to\b|\bkirim ke\b", 4.0),
            (r"\bsupplier\b|\bpemasok\b", 2.5),
            (r"\border date\b|\btgl\.?\s*pesanan\b|\btanggal pesanan\b", 2.5),
        ],
        "receipt": [
            (r"\breceipt\b", 4.5),
            (r"\bstruk\b|\bkwitansi\b|\bnota\b", 5.0),
            (r"\brcpt[-/]\d|\bstr[-/]\d|\brec[-/]\d", 4.0),
            (r"\bmerchant\b|\bkasir\b|\bcashier\b", 3.5),
            (r"\bpayment method\b|\bmetode pembayaran\b|\bpaid by\b|\bpembayaran\b", 3.0),
            (r"\bqris\b|\bdebit\b|\bcredit card\b|\btunai\b|\bcash\b|\be-wallet\b|\bgopay\b|\bovo\b", 2.5),
            (r"\bterima kasih\b|\bthank you\b", 2.0),
            (r"\btransaction date\b|\btgl\.?\s*transaksi\b|\btanggal transaksi\b", 3.0),
        ],
    }

    scores: dict[DocType, float] = {"invoice": 0.0, "purchase_order": 0.0, "receipt": 0.0}
    for doc_type, patterns in signals.items():
        for pattern, weight in patterns:
            if re.search(pattern, text_lower):
                scores[doc_type] += weight
            if re.search(pattern, header_lower):
                scores[doc_type] += weight * 2.0

    best = max(scores.items(), key=lambda item: item[1])
    return best[0] if best[1] > 0 else "invoice"


def _split_segments(lines: list[str]) -> list[str]:
    """Split multi-column lines separated by 2+ spaces into atomic label-value segments."""
    segments: list[str] = []
    for line in lines:
        parts = [p.strip() for p in re.split(r"\s{2,}|\t|\|", line) if p.strip()]
        segments.extend(parts)
    return segments


def _extract_labeled_value(
    lines: list[str],
    tables: list[list[list[str]]],
    label_patterns: list[str],
    allow_next_line: bool = True,
) -> tuple[str | None, str]:
    """Find a value associated with any of the given label regex patterns."""
    segments = _split_segments(lines)

    # 1. Exact same-segment label: value
    for seg in segments:
        for pat in label_patterns:
            m = re.match(rf"^(?:{pat})\s*[:#-]?\s+(.+)$", seg, re.IGNORECASE)
            if m:
                val = m.group(1).strip().strip(":").strip()
                if val and val.lower() not in {"-", "n/a", "none"}:
                    return val, "label"

    # 2. Adjacent columns in structured/reconstructed tables
    for tbl in tables:
        for row in tbl:
            for idx, cell in enumerate(row):
                cleaned = cell.strip()
                for pat in label_patterns:
                    if re.fullmatch(rf"(?:{pat})\s*[:#-]?", cleaned, re.IGNORECASE):
                        if idx + 1 < len(row):
                            val = row[idx + 1].strip()
                            if val and val.lower() not in {"-", "n/a", "none"}:
                                return val, "table"
                    else:
                        m = re.match(rf"^(?:{pat})\s*[:#-]?\s+(.+)$", cleaned, re.IGNORECASE)
                        if m:
                            val = m.group(1).strip()
                            if val and val.lower() not in {"-", "n/a", "none"}:
                                return val, "table"

    # 3. Label alone on one line, value on the next line
    if allow_next_line:
        for idx, line in enumerate(lines[:-1]):
            for pat in label_patterns:
                if re.fullmatch(rf"(?:{pat})\s*[:#-]?", line.strip(), re.IGNORECASE):
                    nxt = lines[idx + 1].strip()
                    if nxt and ":" not in nxt[:22]:
                        return nxt, "label_next_line"

    return None, "missing"


def _extract_money_on_label(lines: list[str], label_patterns: list[str], exclude_patterns: list[str] | None = None) -> tuple[float | None, str]:
    """Extract the rightmost monetary amount from a line matching one of the summary labels."""
    exclude_patterns = exclude_patterns or []
    for line in reversed(lines):
        if any(re.search(ex, line, re.IGNORECASE) for ex in exclude_patterns):
            continue
        for pat in label_patterns:
            if re.search(rf"(?:^|\s|\|)(?:{pat})\b", line, re.IGNORECASE):
                # Remove percentage tokens like "11%" or "(10%)" so tax rate isn't confused with tax amount
                scrubbed = re.sub(r"\(?\s*\d+(?:[.,]\d+)?\s*%\s*\)?", " ", line)
                # Prefer tokens after the matched label
                m = re.search(pat, scrubbed, re.IGNORECASE)
                tail = scrubbed[m.end() :] if m else scrubbed
                tokens = [t.strip() for t in MONEY_TOKEN_PATTERN.findall(tail) if t.strip()]
                for tok in reversed(tokens):
                    parsed = parse_money(tok)
                    if parsed is not None:
                        return parsed, "summary"
    return None, "missing"


def _is_summary_row(text: str) -> bool:
    low = text.lower().strip()
    if not low:
        return True
    if low in {"total", "subtotal", "tax", "ppn", "vat"}:
        return True
    return any(re.search(rf"\b{re.escape(kw)}\b", low) for kw in SUMMARY_ROW_KEYWORDS)


def _is_header_row(cells: list[str]) -> bool:
    joined = " ".join(c.lower() for c in cells)
    hits = sum(1 for kw in TABLE_HEADER_KEYWORDS if kw in joined)
    return hits >= 2


def _parse_quantity(raw: str) -> float | None:
    cleaned = re.sub(r"(?i)\b(?:pcs|unit|units|box|buah|set|hrs|hours|jam|kg|pax|cup|cups|ea|x)\b", "", raw).strip()
    if not cleaned:
        return None
    val = parse_money(cleaned)
    if val is None or val <= 0:
        return None
    return val


def _extract_items_from_table(table: list[list[str]]) -> list[LineItem]:
    """Extract structured line items from a single tabular matrix."""
    if not table or len(table) < 2:
        return []

    header_idx: int | None = None
    col_map: dict[str, int] = {}

    for r_idx, row in enumerate(table):
        if _is_header_row(row):
            header_idx = r_idx
            for c_idx, cell in enumerate(row):
                low = cell.lower()
                if any(k in low for k in ("desc", "desk", "uraian", "item", "barang", "produk", "menu", "service")) and "desc" not in col_map:
                    col_map["desc"] = c_idx
                elif any(k in low for k in ("qty", "quantity", "jumlah")) and "total" not in low and "harga" not in low and "qty" not in col_map:
                    col_map["qty"] = c_idx
                elif any(k in low for k in ("unit price", "harga satuan", "price", "harga", "rate")) and "total" not in low and "price" not in col_map:
                    col_map["price"] = c_idx
                elif any(k in low for k in ("line total", "total", "amount", "subtotal", "jumlah harga")):
                    col_map["total"] = c_idx
            break

    items: list[LineItem] = []
    start_row = (header_idx + 1) if header_idx is not None else 0

    for row in table[start_row:]:
        non_empty = [c.strip() for c in row if c and c.strip()]
        if len(non_empty) < 3:
            continue
        first_joined = " ".join(non_empty[:2])
        if _is_header_row(non_empty) or _is_summary_row(first_joined):
            # Stop once we hit summary rows below a detected table header
            if header_idx is not None and _is_summary_row(non_empty[0]):
                break
            continue

        # Attempt mapped column extraction first
        if {"desc", "qty", "price", "total"} <= col_map.keys() and max(col_map.values()) < len(row):
            desc = row[col_map["desc"]].strip()
            qty = _parse_quantity(row[col_map["qty"]])
            unit_price = parse_money(row[col_map["price"]])
            line_total = parse_money(row[col_map["total"]])
            if desc and not _is_summary_row(desc) and qty is not None and unit_price is not None and line_total is not None:
                items.append(LineItem(description=desc, quantity=qty, unit_price=unit_price, line_total=line_total))
                continue

        # Heuristic positional extraction for rows like [No?, Desc, Qty, UnitPrice, LineTotal]
        candidate = list(non_empty)
        if len(candidate) >= 5 and re.fullmatch(r"\d+\.?", candidate[0]):
            candidate = candidate[1:]
        if len(candidate) == 4:
            desc, qty_s, price_s, total_s = candidate
            qty = _parse_quantity(qty_s)
            unit_price = parse_money(price_s)
            line_total = parse_money(total_s)
            if (
                desc
                and re.search(r"[A-Za-z]", desc)
                and not _is_summary_row(desc)
                and ":" not in desc
                and qty is not None
                and unit_price is not None
                and line_total is not None
            ):
                items.append(LineItem(description=desc, quantity=qty, unit_price=unit_price, line_total=line_total))

    return items


def _extract_items_from_lines(lines: list[str]) -> list[LineItem]:
    """Fallback regex line-item extractor for receipts and plain-text documents."""
    items: list[LineItem] = []

    # Pattern A: "2 x Kopi Susu Aren @ Rp 25.000  Rp 50.000" or "2x Americano @ 20.000 = 40.000"
    pat_a = re.compile(
        r"^(?:\d+\.\s*)?(\d+(?:\.\d+)?)\s*[xX]\s+(.+?)\s+@\s*([A-Za-z$€£.\s]*[\d.,]+(?:,-)?)\s*(?:=|\s{2,}|\s+)\s*([A-Za-z$€£.\s]*[\d.,]+(?:,-)?)$"
    )
    # Pattern B: "Kopi Susu Aren  2 x Rp 25.000  Rp 50.000"
    pat_b = re.compile(
        r"^(?:\d+\.\s*)?(.+?)\s{2,}(\d+(?:\.\d+)?)\s*[xX@]\s*([A-Za-z$€£.\s]*[\d.,]+(?:,-)?)\s{2,}([A-Za-z$€£.\s]*[\d.,]+(?:,-)?)$"
    )
    # Pattern C: "Description  Qty  UnitPrice  LineTotal" (wide-space separated)
    pat_c = re.compile(
        r"^(?:\d+\s{2,})?(.+?)\s{2,}(\d+(?:\.\d+)?(?:\s*(?:pcs|unit|box|hrs|set|pax|cup|ea))?)\s{2,}([A-Za-z$€£.\s]*[\d.,]+(?:,-)?)\s{2,}([A-Za-z$€£.\s]*[\d.,]+(?:,-)?)$",
        re.IGNORECASE,
    )

    for line in lines:
        if _is_summary_row(line) or _is_header_row([line]):
            continue

        m = pat_a.match(line)
        if m:
            qty = _parse_quantity(m.group(1))
            desc = m.group(2).strip()
            price = parse_money(m.group(3))
            total = parse_money(m.group(4))
            if desc and qty and price is not None and total is not None and not _is_summary_row(desc):
                items.append(LineItem(description=desc, quantity=qty, unit_price=price, line_total=total))
                continue

        m = pat_b.match(line) or pat_c.match(line)
        if m:
            desc = m.group(1).strip()
            qty = _parse_quantity(m.group(2))
            price = parse_money(m.group(3))
            total = parse_money(m.group(4))
            if (
                desc
                and re.search(r"[A-Za-z]", desc)
                and ":" not in desc
                and qty
                and price is not None
                and total is not None
                and not _is_summary_row(desc)
            ):
                items.append(LineItem(description=desc, quantity=qty, unit_price=price, line_total=total))

    return items


def _extract_line_items(doc: IngestedDocument) -> tuple[list[LineItem], str]:
    """Extract line items using structured tables first, then layout rows and regex fallback."""
    for idx, tbl in enumerate(doc.tables):
        items = _extract_items_from_table(tbl)
        if items:
            source = "table" if idx < len(doc.tables) - 1 or doc.source_format != "pdf" else "layout_table"
            return items, source

    regex_items = _extract_items_from_lines(doc.lines)
    if regex_items:
        return regex_items, "regex"

    return [], "missing"


def _infer_top_entity_name(lines: list[str]) -> str | None:
    """Infer vendor/merchant name from the header block when no explicit label is present."""
    for line in lines[:5]:
        seg = _split_segments([line])[0]
        low = seg.lower().strip()
        if not seg or low in DOC_TITLE_NOISE:
            continue
        if ":" in seg or re.search(r"\b(?:jl\.|street|st\.|ave|tel|phone|npwp|email|www\.|http)\b", low):
            continue
        if re.search(r"[A-Za-z]{3,}", seg) and len(seg) <= 80:
            return seg
    return None


def _extract_invoice_local(doc: IngestedDocument) -> RawExtractionOutput:
    lines = doc.lines
    tables = doc.tables
    sources: dict[str, str] = {}
    raw_fields: dict[str, Any] = {}

    currency = detect_currency(doc.raw_text)
    sources["currency"] = "regex"

    inv_no, src = _extract_labeled_value(
        lines,
        tables,
        [r"invoice\s*(?:no\.?|number|#)", r"no\.?\s*invoice", r"no\.?\s*faktur", r"faktur\s*(?:no\.?|nomor)", r"nomor\s*faktur", r"nomor\s*invoice"],
        allow_next_line=False,
    )
    if not inv_no:
        m = re.search(r"\b((?:INV|FKT|FAK)[-/][A-Z0-9-]{3,})\b", doc.raw_text)
        if m:
            inv_no, src = m.group(1), "regex"
    sources["invoice_number"] = src

    issue_raw, src = _extract_labeled_value(
        lines,
        tables,
        [r"issue\s*date", r"invoice\s*date", r"tanggal\s*faktur", r"tanggal\s*invoice", r"tgl\.?\s*faktur", r"tgl\.?\s*invoice", r"tanggal", r"date"],
        allow_next_line=False,
    )
    raw_fields["issue_date"] = issue_raw
    issue_date = parse_date(issue_raw) if issue_raw else None
    sources["issue_date"] = src if issue_date else ("unparsed" if issue_raw else "missing")

    due_raw, src = _extract_labeled_value(
        lines,
        tables,
        [r"due\s*date", r"payment\s*due", r"jatuh\s*tempo", r"tgl\.?\s*jatuh\s*tempo", r"tanggal\s*jatuh\s*tempo"],
        allow_next_line=False,
    )
    raw_fields["due_date"] = due_raw
    due_date = parse_date(due_raw) if due_raw else None
    sources["due_date"] = src if due_date else ("unparsed" if due_raw else "missing")

    vendor_name, src = _extract_labeled_value(
        lines,
        tables,
        [r"vendor(?:\s*name)?", r"seller", r"from", r"penjual", r"pengirim", r"diterbitkan\s*oleh", r"issued\s*by"],
    )
    if not vendor_name:
        inferred = _infer_top_entity_name(lines)
        if inferred:
            vendor_name, src = inferred, "header"
    sources["vendor_name"] = src

    npwp_raw, src = _extract_labeled_value(lines, tables, [r"vendor\s*npwp", r"npwp\s*penjual", r"npwp"], allow_next_line=False)
    vendor_npwp = None
    if npwp_raw:
        m = NPWP_PATTERN.search(npwp_raw) or NPWP_16_PATTERN.search(npwp_raw)
        vendor_npwp = m.group(0) if m else npwp_raw
    else:
        m = NPWP_PATTERN.search(doc.raw_text)
        if m:
            vendor_npwp, src = m.group(0), "regex"
    sources["vendor_npwp"] = src

    buyer_name, src = _extract_labeled_value(
        lines,
        tables,
        [r"bill\s*to", r"billed\s*to", r"buyer(?:\s*name)?", r"customer(?:\s*name)?", r"client", r"tagihan\s*kepada", r"kepada\s*yth\.?", r"pembeli", r"pelanggan"],
    )
    sources["buyer_name"] = src

    line_items, item_src = _extract_line_items(doc)
    sources["line_items"] = item_src

    subtotal, src = _extract_money_on_label(lines, [r"subtotal", r"sub\s*-?\s*total", r"dpp", r"jumlah\s*sebelum\s*pajak"])
    sources["subtotal"] = src

    tax_amount, src = _extract_money_on_label(
        lines,
        [r"ppn", r"vat", r"sales\s*tax", r"tax(?:\s*amount)?", r"pajak"],
        exclude_patterns=[r"tax\s*invoice", r"faktur\s*pajak", r"npwp", r"sebelum\s*pajak"],
    )
    sources["tax_amount"] = src

    total_amount, src = _extract_money_on_label(
        lines,
        [r"grand\s*total", r"total\s*amount", r"total\s*due", r"amount\s*due", r"total\s*tagihan", r"total\s*bayar", r"total\s*pembayaran", r"total"],
        exclude_patterns=[r"sub\s*-?\s*total", r"line\s*total", r"unit\s*price", r"qty", r"quantity"],
    )
    sources["total_amount"] = src

    data = InvoiceData(
        invoice_number=inv_no,
        issue_date=issue_date or issue_raw,
        due_date=due_date or due_raw,
        vendor_name=vendor_name,
        vendor_npwp=vendor_npwp,
        buyer_name=buyer_name,
        currency=currency,
        line_items=line_items,
        subtotal=subtotal,
        tax_amount=tax_amount,
        total_amount=total_amount,
    )
    return RawExtractionOutput(doc_type="invoice", data=data, field_sources=sources, raw_fields=raw_fields, provider="local")


def _extract_po_local(doc: IngestedDocument) -> RawExtractionOutput:
    lines = doc.lines
    tables = doc.tables
    sources: dict[str, str] = {}
    raw_fields: dict[str, Any] = {}

    currency = detect_currency(doc.raw_text)
    sources["currency"] = "regex"

    po_number, src = _extract_labeled_value(
        lines,
        tables,
        [r"po\s*(?:no\.?|number|#)", r"purchase\s*order\s*(?:no\.?|number|#)", r"no\.?\s*po", r"nomor\s*po", r"no\.?\s*pesanan"],
        allow_next_line=False,
    )
    if not po_number:
        m = re.search(r"\b(PO[-/][A-Z0-9-]{3,})\b", doc.raw_text)
        if m:
            po_number, src = m.group(1), "regex"
    sources["po_number"] = src

    order_raw, src = _extract_labeled_value(
        lines,
        tables,
        [r"order\s*date", r"po\s*date", r"tanggal\s*pesanan", r"tgl\.?\s*pesanan", r"tanggal\s*po", r"tanggal", r"date"],
        allow_next_line=False,
    )
    raw_fields["order_date"] = order_raw
    order_date = parse_date(order_raw) if order_raw else None
    sources["order_date"] = src if order_date else ("unparsed" if order_raw else "missing")

    supplier_name, src = _extract_labeled_value(
        lines,
        tables,
        [r"supplier(?:\s*name)?", r"vendor(?:\s*name)?", r"pemasok", r"kepada\s*pemasok"],
    )
    sources["supplier_name"] = src

    buyer_name, src = _extract_labeled_value(
        lines,
        tables,
        [r"buyer(?:\s*name)?", r"purchaser", r"ordered\s*by", r"company", r"pembeli", r"pemesan", r"dari"],
    )
    if not buyer_name:
        inferred = _infer_top_entity_name(lines)
        if inferred and inferred != supplier_name:
            buyer_name, src = inferred, "header"
    sources["buyer_name"] = src

    delivery_address, src = _extract_labeled_value(
        lines,
        tables,
        [r"delivery\s*address", r"ship\s*to", r"shipping\s*address", r"alamat\s*pengiriman", r"kirim\s*ke"],
    )
    sources["delivery_address"] = src

    line_items, item_src = _extract_line_items(doc)
    sources["line_items"] = item_src

    total_amount, src = _extract_money_on_label(
        lines,
        [r"grand\s*total", r"total\s*amount", r"total\s*order", r"total\s*po", r"total\s*pesanan", r"total"],
        exclude_patterns=[r"sub\s*-?\s*total", r"line\s*total", r"unit\s*price", r"qty"],
    )
    sources["total_amount"] = src

    data = PurchaseOrderData(
        po_number=po_number,
        order_date=order_date or order_raw,
        supplier_name=supplier_name,
        buyer_name=buyer_name,
        delivery_address=delivery_address,
        currency=currency,
        line_items=line_items,
        total_amount=total_amount,
    )
    return RawExtractionOutput(doc_type="purchase_order", data=data, field_sources=sources, raw_fields=raw_fields, provider="local")


def _extract_receipt_local(doc: IngestedDocument) -> RawExtractionOutput:
    lines = doc.lines
    tables = doc.tables
    sources: dict[str, str] = {}
    raw_fields: dict[str, Any] = {}

    currency = detect_currency(doc.raw_text)
    sources["currency"] = "regex"

    merchant_name, src = _extract_labeled_value(
        lines,
        tables,
        [r"merchant(?:\s*name)?", r"store(?:\s*name)?", r"toko", r"outlet"],
    )
    if not merchant_name:
        inferred = _infer_top_entity_name(lines)
        if inferred:
            merchant_name, src = inferred, "header"
    sources["merchant_name"] = src

    tx_raw, src = _extract_labeled_value(
        lines,
        tables,
        [r"transaction\s*date", r"receipt\s*date", r"date", r"tgl\.?\s*transaksi", r"tanggal\s*transaksi", r"tanggal", r"tgl\.?"],
        allow_next_line=False,
    )
    raw_fields["transaction_date"] = tx_raw
    tx_date = parse_date(tx_raw) if tx_raw else None
    sources["transaction_date"] = src if tx_date else ("unparsed" if tx_raw else "missing")

    receipt_number, src = _extract_labeled_value(
        lines,
        tables,
        [r"receipt\s*(?:no\.?|number|#)", r"no\.?\s*struk", r"nomor\s*struk", r"no\.?\s*nota", r"no\.?\s*kwitansi", r"trx\s*(?:id|no\.?)"],
        allow_next_line=False,
    )
    if not receipt_number:
        m = re.search(r"\b((?:RCPT|STR|REC|TRX)[-/][A-Z0-9-]{3,})\b", doc.raw_text)
        if m:
            receipt_number, src = m.group(1), "regex"
    sources["receipt_number"] = src

    payment_method, src = _extract_labeled_value(
        lines,
        tables,
        [r"payment\s*method", r"paid\s*by", r"payment", r"metode\s*pembayaran", r"pembayaran", r"cara\s*bayar"],
        allow_next_line=False,
    )
    sources["payment_method"] = src

    line_items, item_src = _extract_line_items(doc)
    sources["line_items"] = item_src

    tax_amount, src = _extract_money_on_label(
        lines,
        [r"ppn", r"pb1", r"tax", r"vat", r"pajak"],
        exclude_patterns=[r"tax\s*invoice", r"npwp"],
    )
    sources["tax_amount"] = src

    total_amount, src = _extract_money_on_label(
        lines,
        [r"grand\s*total", r"total\s*amount", r"total\s*bayar", r"total\s*pembayaran", r"total\s*paid", r"total"],
        exclude_patterns=[r"sub\s*-?\s*total", r"line\s*total", r"unit\s*price", r"qty"],
    )
    sources["total_amount"] = src

    data = ReceiptData(
        merchant_name=merchant_name,
        transaction_date=tx_date or tx_raw,
        receipt_number=receipt_number,
        payment_method=payment_method,
        currency=currency,
        line_items=line_items,
        tax_amount=tax_amount,
        total_amount=total_amount,
    )
    return RawExtractionOutput(doc_type="receipt", data=data, field_sources=sources, raw_fields=raw_fields, provider="local")


def _strip_json_fences(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _build_llm_prompt(doc: IngestedDocument, doc_type: DocType) -> str:
    schema_cls = schema_for_doc_type(doc_type)
    schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
    return (
        f"You are a precision financial document extraction engine.\n"
        f"Extract structured data from this {doc_type} into STRICT JSON matching the schema below.\n"
        f"Rules:\n"
        f"- Normalize all dates to YYYY-MM-DD.\n"
        f"- Parse all Indonesian (Rp 1.250.000) and English ($1,250.00) numbers into pure floats.\n"
        f"- Preserve exact extracted values from the document (do not invent or alter mismatched totals).\n"
        f"- Return ONLY valid JSON with no markdown commentary.\n\n"
        f"Target JSON Schema:\n{schema_json}\n\n"
        f"Document Text:\n{doc.raw_text or '[See attached document image]'}"
    )


def _extract_via_ollama(doc: IngestedDocument, doc_type: DocType, settings: Settings) -> RawExtractionOutput:
    prompt = _build_llm_prompt(doc, doc_type)
    user_msg: dict[str, Any] = {"role": "user", "content": prompt}
    if doc.page_images_b64:
        user_msg["images"] = doc.page_images_b64[:2]

    payload = {
        "model": settings.model,
        "messages": [user_msg],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }
    url = f"{settings.ollama_base_url}/api/chat"
    with httpx.Client(timeout=settings.http_timeout_seconds) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]

    parsed = json.loads(_strip_json_fences(content))
    model_cls = schema_for_doc_type(doc_type)
    validated = model_cls.model_validate(parsed)
    sources = {k: "llm:ollama" for k in model_cls.model_fields}
    return RawExtractionOutput(doc_type=doc_type, data=validated, field_sources=sources, provider="ollama")


def _extract_via_gemini(doc: IngestedDocument, doc_type: DocType, settings: Settings) -> RawExtractionOutput:
    if not settings.gemini_api_key:
        raise ValueError("DOCINTEL_GEMINI_API_KEY is required when provider='gemini'.")

    prompt = _build_llm_prompt(doc, doc_type)
    parts: list[dict[str, Any]] = [{"text": prompt}]
    for img_b64 in doc.page_images_b64[:2]:
        parts.append({"inlineData": {"mimeType": "image/png", "data": img_b64}})

    model_name = settings.model if "gemini" in settings.model.lower() else "gemini-2.0-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
    }
    with httpx.Client(timeout=settings.http_timeout_seconds) as client:
        resp = client.post(url, params={"key": settings.gemini_api_key}, json=payload)
        resp.raise_for_status()
        body = resp.json()
        content = body["candidates"][0]["content"]["parts"][0]["text"]

    parsed = json.loads(_strip_json_fences(content))
    model_cls = schema_for_doc_type(doc_type)
    validated = model_cls.model_validate(parsed)
    sources = {k: "llm:gemini" for k in model_cls.model_fields}
    return RawExtractionOutput(doc_type=doc_type, data=validated, field_sources=sources, provider="gemini")


def _extract_via_openai(doc: IngestedDocument, doc_type: DocType, settings: Settings) -> RawExtractionOutput:
    if not settings.openai_api_key:
        raise ValueError("DOCINTEL_OPENAI_API_KEY is required when provider='openai'.")

    prompt = _build_llm_prompt(doc, doc_type)
    content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img_b64 in doc.page_images_b64[:2]:
        content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

    model_name = settings.model if any(k in settings.model.lower() for k in ("gpt", "o1", "o3", "o4")) else "gpt-4o-mini"
    url = f"{settings.openai_base_url}/chat/completions"
    payload = {
        "model": model_name,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content_parts}],
    }
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    with httpx.Client(timeout=settings.http_timeout_seconds) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

    parsed = json.loads(_strip_json_fences(content))
    model_cls = schema_for_doc_type(doc_type)
    validated = model_cls.model_validate(parsed)
    sources = {k: "llm:openai" for k in model_cls.model_fields}
    return RawExtractionOutput(doc_type=doc_type, data=validated, field_sources=sources, provider="openai")


class DocumentExtractor:
    """Hybrid document extraction engine supporting deterministic local parsing and Vision-LLMs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def extract(
        self,
        doc: IngestedDocument,
        doc_type: DocTypeRequest = "auto",
        provider_override: str | None = None,
    ) -> RawExtractionOutput:
        """Classify (if auto) and extract structured schema data from an IngestedDocument."""
        resolved_type: DocType = classify_document(doc) if doc_type == "auto" else doc_type
        provider = (provider_override or self.settings.provider).lower()

        if provider == "local":
            if resolved_type == "invoice":
                return _extract_invoice_local(doc)
            if resolved_type == "purchase_order":
                return _extract_po_local(doc)
            return _extract_receipt_local(doc)

        if provider == "ollama":
            return _extract_via_ollama(doc, resolved_type, self.settings)
        if provider == "gemini":
            return _extract_via_gemini(doc, resolved_type, self.settings)
        if provider == "openai":
            return _extract_via_openai(doc, resolved_type, self.settings)

        raise ValueError(f"Unsupported provider '{provider}'. Choose local, ollama, gemini, or openai.")
