"""FastAPI application for DocIntel Extract & HITL Review Studio."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from docintel import __version__
from docintel.config import Settings, get_settings
from docintel.ingestion import sanitize_filename
from docintel.pipeline import DocumentIntelligencePipeline
from docintel.review_store import ExportFormat, ReviewStore
from docintel.schemas import (
    BatchExtractResponse,
    BatchTextExtractRequest,
    DocTypeRequest,
    ExtractionResult,
    HealthResponse,
    ReviewSubmission,
    TextExtractRequest,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _resolve_samples_dir(settings: Settings) -> Path:
    if settings.samples_dir.is_dir():
        return settings.samples_dir
    # Fallback relative to repository root when launched from subdirectories
    repo_root_samples = Path(__file__).resolve().parents[2] / "data" / "samples"
    if repo_root_samples.is_dir():
        return repo_root_samples
    return settings.samples_dir


async def _read_upload_limited(uploaded: UploadFile, max_bytes: int) -> bytes:
    """Read an uploaded file in chunks while enforcing a hard upper size limit."""
    chunks: list[bytes] = []
    total = 0
    chunk_size = 1024 * 1024
    while True:
        chunk = await uploaded.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            limit_mb = max_bytes / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded file exceeds maximum allowed size of {limit_mb:.1f} MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _handle_extraction_error(exc: Exception) -> HTTPException:
    """Map ingestion/validation/provider exceptions to clean 4xx/502 responses without leaking internals."""
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, ValidationError):
        return HTTPException(status_code=422, detail=exc.errors())
    if isinstance(exc, httpx.HTTPStatusError):
        return HTTPException(
            status_code=502,
            detail=f"Upstream Vision-LLM provider returned HTTP {exc.response.status_code}.",
        )
    if isinstance(exc, httpx.HTTPError):
        return HTTPException(
            status_code=502,
            detail="Failed to communicate with upstream Vision-LLM provider.",
        )
    return HTTPException(status_code=500, detail="Internal document extraction error.")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the DocIntel Extract FastAPI application."""
    runtime_settings = settings or get_settings()
    store = ReviewStore(runtime_settings.db_path, auto_init=False)
    pipeline = DocumentIntelligencePipeline(runtime_settings, store=store)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        runtime_settings.upload_dir.mkdir(parents=True, exist_ok=True)
        store.init_schema()
        yield

    app = FastAPI(
        title="DocIntel Extract API",
        description=(
            "Multimodal Document Intelligence Pipeline converting unstructured PDFs, invoices, "
            "purchase orders, and receipts into validated Pydantic JSON with per-field confidence "
            "scoring, arithmetic reconciliation, and Human-in-the-Loop (HITL) review."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    allow_credentials = "*" not in runtime_settings.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = runtime_settings
    app.state.store = store
    app.state.pipeline = pipeline

    @app.get("/", include_in_schema=False)
    async def serve_studio_ui() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            raise HTTPException(status_code=404, detail="Studio UI not found.")
        return FileResponse(index_path, media_type="text/html; charset=utf-8")

    @app.get("/health", response_model=HealthResponse, tags=["System"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            provider=runtime_settings.provider,
            review_confidence_threshold=runtime_settings.review_confidence_threshold,
            jobs_summary=store.get_status_counts(),
        )

    @app.get("/v1/samples", tags=["Samples"])
    async def list_samples() -> dict[str, list[dict[str, str | int]]]:
        samples_dir = _resolve_samples_dir(runtime_settings)
        if not samples_dir.is_dir():
            return {"samples": []}
        entries: list[dict[str, str | int]] = []
        for pdf_path in sorted(samples_dir.glob("*.pdf")):
            entries.append(
                {
                    "filename": pdf_path.name,
                    "size_bytes": pdf_path.stat().st_size,
                }
            )
        return {"samples": entries}

    @app.post("/v1/samples/{filename}/extract", response_model=ExtractionResult, tags=["Samples"])
    async def extract_sample(
        filename: str,
        doc_type: DocTypeRequest = Query(default="auto"),
        provider: str | None = Query(default=None),
    ) -> ExtractionResult:
        samples_dir = _resolve_samples_dir(runtime_settings).resolve()
        safe_name = sanitize_filename(filename, default="")
        target = (samples_dir / safe_name).resolve()
        if (
            not safe_name
            or not target.is_relative_to(samples_dir)
            or not target.is_file()
            or target.suffix.lower() != ".pdf"
        ):
            raise HTTPException(status_code=404, detail=f"Sample PDF '{safe_name or filename}' not found.")
        try:
            return pipeline.process_file(target, doc_type=doc_type, provider_override=provider)
        except Exception as exc:
            raise _handle_extraction_error(exc) from exc

    @app.post(
        "/v1/extract",
        response_model=ExtractionResult,
        tags=["Extraction"],
        summary="Extract structured JSON from a PDF/image/text upload or JSON text payload",
    )
    async def extract_document(request: Request) -> ExtractionResult:
        content_type = (request.headers.get("content-type") or "").lower()

        if "multipart/form-data" in content_type:
            form = await request.form()
            uploaded = form.get("file")
            if uploaded is None or not hasattr(uploaded, "read"):
                raise HTTPException(status_code=422, detail="Multipart request must include a 'file' field.")

            doc_type_raw = str(form.get("doc_type") or "auto").lower()
            if doc_type_raw not in {"auto", "invoice", "purchase_order", "receipt"}:
                raise HTTPException(status_code=422, detail=f"Invalid doc_type '{doc_type_raw}'.")
            provider_raw = form.get("provider")
            provider_override = str(provider_raw).strip() if provider_raw else None

            raw_bytes = await _read_upload_limited(uploaded, runtime_settings.max_upload_bytes)
            if not raw_bytes:
                raise HTTPException(status_code=400, detail="Uploaded file is empty.")

            filename = sanitize_filename(getattr(uploaded, "filename", None), default="upload.pdf")
            file_mime = getattr(uploaded, "content_type", None)
            try:
                return pipeline.process_bytes(
                    content=raw_bytes,
                    filename=filename,
                    content_type=file_mime,
                    doc_type=doc_type_raw,  # type: ignore[arg-type]
                    provider_override=provider_override,
                )
            except Exception as exc:
                raise _handle_extraction_error(exc) from exc

        # JSON text payload
        try:
            body = await request.json()
            payload = TextExtractRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Send either multipart/form-data with 'file' or JSON matching TextExtractRequest.",
            ) from exc

        if len(payload.text.encode("utf-8")) > runtime_settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Text payload exceeds maximum allowed size.")

        try:
            return pipeline.process_text(
                text=payload.text,
                filename=sanitize_filename(payload.filename, default="document.txt"),
                doc_type=payload.doc_type,
                provider_override=payload.provider,
            )
        except Exception as exc:
            raise _handle_extraction_error(exc) from exc

    @app.post(
        "/v1/extract/batch",
        response_model=BatchExtractResponse,
        tags=["Extraction"],
        summary="Batch extract multiple uploaded files or JSON text documents",
    )
    async def extract_batch(request: Request) -> BatchExtractResponse:
        content_type = (request.headers.get("content-type") or "").lower()
        results: list[ExtractionResult] = []

        if "multipart/form-data" in content_type:
            form = await request.form()
            doc_type_raw = str(form.get("doc_type") or "auto").lower()
            if doc_type_raw not in {"auto", "invoice", "purchase_order", "receipt"}:
                raise HTTPException(status_code=422, detail=f"Invalid doc_type '{doc_type_raw}'.")
            provider_raw = form.get("provider")
            provider_override = str(provider_raw).strip() if provider_raw else None

            files = [item for item in form.getlist("files") if isinstance(item, UploadFile) or hasattr(item, "read")]
            if not files:
                raise HTTPException(status_code=422, detail="Multipart batch requires one or more 'files' parts.")
            if len(files) > runtime_settings.max_batch_files:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch exceeds maximum limit of {runtime_settings.max_batch_files} files.",
                )

            for f in files:
                data = await _read_upload_limited(f, runtime_settings.max_upload_bytes)
                if not data:
                    continue
                try:
                    res = pipeline.process_bytes(
                        content=data,
                        filename=sanitize_filename(getattr(f, "filename", None), default="batch_doc.pdf"),
                        content_type=getattr(f, "content_type", None),
                        doc_type=doc_type_raw,  # type: ignore[arg-type]
                        provider_override=provider_override,
                    )
                except Exception as exc:
                    raise _handle_extraction_error(exc) from exc
                results.append(res)
        else:
            try:
                body = await request.json()
                batch_req = BatchTextExtractRequest.model_validate(body)
            except ValidationError as exc:
                raise HTTPException(status_code=422, detail=exc.errors()) from exc
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Invalid batch extraction payload.") from exc

            if len(batch_req.documents) > runtime_settings.max_batch_files:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch exceeds maximum limit of {runtime_settings.max_batch_files} documents.",
                )

            for item in batch_req.documents:
                if len(item.text.encode("utf-8")) > runtime_settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Text payload exceeds maximum allowed size.")
                try:
                    results.append(
                        pipeline.process_text(
                            text=item.text,
                            filename=sanitize_filename(item.filename, default="document.txt"),
                            doc_type=item.doc_type,
                            provider_override=item.provider,
                        )
                    )
                except Exception as exc:
                    raise _handle_extraction_error(exc) from exc

        approved = sum(1 for r in results if r.status == "APPROVED")
        needs_review = sum(1 for r in results if r.status == "NEEDS_REVIEW")
        return BatchExtractResponse(
            total=len(results),
            approved=approved,
            needs_review=needs_review,
            results=results,
        )

    @app.get("/v1/jobs", response_model=list[ExtractionResult], tags=["HITL Review Queue"])
    async def list_jobs(
        status: Literal["APPROVED", "NEEDS_REVIEW", "REVIEWED"] | None = Query(default=None),
        doc_type: Literal["invoice", "purchase_order", "receipt"] | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> list[ExtractionResult]:
        return store.list_jobs(status=status, doc_type=doc_type, limit=limit, offset=offset)

    @app.get("/v1/jobs/{job_id}", response_model=ExtractionResult, tags=["HITL Review Queue"])
    async def get_job(job_id: str) -> ExtractionResult:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        return job

    @app.post("/v1/jobs/{job_id}/review", response_model=ExtractionResult, tags=["HITL Review Queue"])
    async def review_job(job_id: str, submission: ReviewSubmission) -> ExtractionResult:
        try:
            return pipeline.submit_review(job_id, submission)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

    @app.get("/v1/export", tags=["Export"])
    async def export_jobs(
        format: ExportFormat = Query(default="json"),
        include_needs_review: bool = Query(default=False),
    ) -> Response:
        statuses = ["APPROVED", "REVIEWED", "NEEDS_REVIEW"] if include_needs_review else ["APPROVED", "REVIEWED"]
        content, media_type = store.export_records(fmt=format, statuses=statuses)
        filename = f"docintel_export.{format}"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return Response(content=content, media_type=media_type, headers=headers)

    return app


app = create_app()
