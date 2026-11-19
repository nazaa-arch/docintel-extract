"""Multimodal document ingestion for PDFs (PyMuPDF + pdfplumber), images (Pillow), and text."""

from __future__ import annotations

import base64
import io
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

SourceFormat = Literal["pdf", "image", "text"]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log"}


@dataclass
class IngestedDocument:
    """Normalized representation of an ingested PDF, image, or text document."""

    filename: str
    mime_type: str
    source_format: SourceFormat
    raw_text: str
    lines: list[str] = field(default_factory=list)
    blocks: list[dict[str, Any]] = field(default_factory=list)
    tables: list[list[list[str]]] = field(default_factory=list)
    page_count: int = 1
    page_images_b64: list[str] = field(default_factory=list)
    image_metadata: dict[str, Any] | None = None


def _clean_cell(cell: Any) -> str:
    if cell is None:
        return ""
    return " ".join(str(cell).replace("\n", " ").split()).strip()


def _reconstruct_visual_lines(page: fitz.Page, y_tolerance: float = 3.5) -> tuple[list[str], list[list[str]]]:
    """Group PyMuPDF word bounding boxes by visual row for borderless tables & multi-column layouts."""
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)
    if not words:
        return [], []

    # Sort primarily by vertical midpoint, then left-to-right x0
    sorted_words = sorted(words, key=lambda w: (round((w[1] + w[3]) / 2.0, 1), w[0]))
    rows: list[list[tuple[float, float, float, float, str]]] = []

    for w in sorted_words:
        x0, y0, x1, y1, text = float(w[0]), float(w[1]), float(w[2]), float(w[3]), str(w[4])
        y_mid = (y0 + y1) / 2.0
        if not rows:
            rows.append([(x0, y_mid, x1, y1, text)])
            continue
        prev_y = sum(item[1] for item in rows[-1]) / len(rows[-1])
        if abs(y_mid - prev_y) <= y_tolerance:
            rows[-1].append((x0, y_mid, x1, y1, text))
        else:
            rows.append([(x0, y_mid, x1, y1, text)])

    visual_lines: list[str] = []
    columnar_rows: list[list[str]] = []

    for row in rows:
        row.sort(key=lambda item: item[0])
        cells: list[str] = []
        current_tokens: list[str] = [row[0][4]]
        prev_x1 = row[0][2]

        for item in row[1:]:
            gap = item[0] - prev_x1
            # Horizontal gap > 14pt usually separates table columns or label/value columns
            if gap > 14.0:
                cells.append(" ".join(current_tokens).strip())
                current_tokens = [item[4]]
            else:
                current_tokens.append(item[4])
            prev_x1 = item[2]

        cells.append(" ".join(current_tokens).strip())
        visual_lines.append("  ".join(c for c in cells if c))
        if len(cells) >= 2:
            columnar_rows.append(cells)

    return visual_lines, columnar_rows


def ingest_pdf_bytes(
    content: bytes,
    filename: str = "document.pdf",
    render_images: bool = False,
    max_render_pages: int = 3,
) -> IngestedDocument:
    """Extract layout blocks, ordered lines, tables, and optional page PNGs from PDF bytes."""
    blocks: list[dict[str, Any]] = []
    all_lines: list[str] = []
    tables: list[list[list[str]]] = []
    page_images_b64: list[str] = []
    reconstructed_rows_all: list[list[str]] = []

    with fitz.open(stream=content, filetype="pdf") as pdf_doc:
        page_count = len(pdf_doc)
        for page_idx in range(page_count):
            page = pdf_doc[page_idx]
            page_blocks = page.get_text("blocks")
            page_blocks_sorted = sorted(page_blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))

            for blk in page_blocks_sorted:
                text = str(blk[4]).strip()
                if not text:
                    continue
                blocks.append(
                    {
                        "page": page_idx + 1,
                        "bbox": [round(float(blk[0]), 2), round(float(blk[1]), 2), round(float(blk[2]), 2), round(float(blk[3]), 2)],
                        "text": text,
                    }
                )

            visual_lines, col_rows = _reconstruct_visual_lines(page)
            all_lines.extend(visual_lines)
            if col_rows:
                reconstructed_rows_all.extend(col_rows)

            if render_images and page_idx < max_render_pages:
                pix = page.get_pixmap(dpi=150)
                png_bytes = pix.tobytes("png")
                page_images_b64.append(base64.b64encode(png_bytes).decode("ascii"))

    # Use pdfplumber for bordered/ruled structured table extraction
    try:
        with pdfplumber.open(io.BytesIO(content)) as plumber_pdf:
            for page in plumber_pdf.pages:
                extracted_tables = page.extract_tables() or []
                for tbl in extracted_tables:
                    cleaned_table = [
                        [_clean_cell(cell) for cell in row]
                        for row in tbl
                        if row and any(_clean_cell(cell) for cell in row)
                    ]
                    if cleaned_table:
                        tables.append(cleaned_table)
    except Exception:
        # Fallback cleanly to PyMuPDF reconstructed rows if pdfplumber fails on malformed streams
        pass

    if reconstructed_rows_all:
        tables.append(reconstructed_rows_all)

    raw_text = "\n".join(all_lines).strip()
    return IngestedDocument(
        filename=filename,
        mime_type="application/pdf",
        source_format="pdf",
        raw_text=raw_text,
        lines=all_lines,
        blocks=blocks,
        tables=tables,
        page_count=page_count,
        page_images_b64=page_images_b64,
        image_metadata=None,
    )


def ingest_image_bytes(content: bytes, filename: str = "image.png") -> IngestedDocument:
    """Inspect an image via Pillow and prepare base64 PNG payload for Vision-LLM adapters."""
    with Image.open(io.BytesIO(content)) as img:
        width, height = img.size
        img_format = img.format or "PNG"
        mode = img.mode
        embedded_text = str(img.info.get("docintel_text", "") or "").strip()

        normalized = img.convert("RGB") if mode not in {"RGB", "RGBA", "L"} else img
        out_buf = io.BytesIO()
        normalized.save(out_buf, format="PNG")
        b64_png = base64.b64encode(out_buf.getvalue()).decode("ascii")

    lines = [line.strip() for line in embedded_text.splitlines() if line.strip()]
    mime = mimetypes.guess_type(filename)[0] or f"image/{img_format.lower()}"

    return IngestedDocument(
        filename=filename,
        mime_type=mime,
        source_format="image",
        raw_text=embedded_text,
        lines=lines,
        blocks=[{"page": 1, "bbox": [0, 0, width, height], "text": embedded_text}] if embedded_text else [],
        tables=[],
        page_count=1,
        page_images_b64=[b64_png],
        image_metadata={
            "width": width,
            "height": height,
            "format": img_format,
            "mode": mode,
            "bytes": len(content),
        },
    )


def ingest_text(text: str, filename: str = "document.txt") -> IngestedDocument:
    """Ingest raw UTF-8 text into a normalized IngestedDocument."""
    clean_text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    blocks = [{"page": 1, "bbox": [0, idx * 14, 600, (idx + 1) * 14], "text": line} for idx, line in enumerate(lines)]

    # Detect tabular lines delimited by tabs, pipes, or wide spacing
    table_rows: list[list[str]] = []
    for line in lines:
        if "|" in line:
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 3 and not all(set(p) <= {"-", ":"} for p in parts if p):
                table_rows.append(parts)
        elif "\t" in line:
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if len(parts) >= 3:
                table_rows.append(parts)

    tables = [table_rows] if table_rows else []
    return IngestedDocument(
        filename=filename,
        mime_type="text/plain",
        source_format="text",
        raw_text=clean_text,
        lines=lines,
        blocks=blocks,
        tables=tables,
        page_count=1,
        page_images_b64=[],
        image_metadata=None,
    )


def ingest_bytes(
    content: bytes,
    filename: str,
    content_type: str | None = None,
    render_images: bool = False,
) -> IngestedDocument:
    """Auto-dispatch raw bytes to PDF, Image, or UTF-8 text ingestion."""
    ext = Path(filename).suffix.lower()
    mime = (content_type or mimetypes.guess_type(filename)[0] or "").lower()

    if ext == ".pdf" or mime == "application/pdf" or content.startswith(b"%PDF"):
        return ingest_pdf_bytes(content, filename=filename, render_images=render_images)

    if ext in IMAGE_EXTENSIONS or mime.startswith("image/"):
        return ingest_image_bytes(content, filename=filename)

    if ext in TEXT_EXTENSIONS or mime.startswith("text/") or not ext:
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError:
            decoded = content.decode("latin-1")
        return ingest_text(decoded, filename=filename)

    raise ValueError(f"Unsupported file format for '{filename}' (mime='{mime or 'unknown'}'). Supported: PDF, PNG/JPG/WEBP, TXT.")


def ingest_file(path: Path, render_images: bool = False) -> IngestedDocument:
    """Ingest a document from disk."""
    return ingest_bytes(path.read_bytes(), filename=path.name, render_images=render_images)
