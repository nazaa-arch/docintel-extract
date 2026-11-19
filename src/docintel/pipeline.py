"""End-to-end document intelligence pipeline orchestration."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from docintel.config import Settings
from docintel.extractor import DocumentExtractor
from docintel.ingestion import IngestedDocument, ingest_bytes, ingest_file, ingest_text
from docintel.review_store import ReviewStore
from docintel.schemas import (
    DocTypeRequest,
    ExtractionResult,
    ReviewSubmission,
    schema_for_doc_type,
    utc_now_iso,
)
from docintel.validator import validate_extraction


class DocumentIntelligencePipeline:
    """Coordinates ingestion, hybrid extraction, validation/reconciliation, and HITL persistence."""

    def __init__(self, settings: Settings, store: ReviewStore | None = None) -> None:
        self.settings = settings
        self.extractor = DocumentExtractor(settings)
        self.store = store or ReviewStore(settings.db_path)

    def _run_on_ingested(
        self,
        doc: IngestedDocument,
        doc_type: DocTypeRequest = "auto",
        provider_override: str | None = None,
        persist: bool = True,
        started_at: float | None = None,
    ) -> ExtractionResult:
        t0 = started_at if started_at is not None else time.perf_counter()
        raw = self.extractor.extract(doc, doc_type=doc_type, provider_override=provider_override)
        validation = validate_extraction(
            doc_type=raw.doc_type,
            data=raw.data,
            field_sources=raw.field_sources,
            review_confidence_threshold=self.settings.review_confidence_threshold,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        result = ExtractionResult(
            job_id=f"job_{uuid.uuid4().hex[:12]}",
            filename=doc.filename,
            doc_type=raw.doc_type,
            status=validation.status,
            provider=raw.provider,
            overall_confidence=validation.overall_confidence,
            reconciliation_passed=validation.reconciliation_passed,
            reconciliation_notes=validation.reconciliation_notes,
            data=raw.data,
            field_confidences=validation.field_confidences,
            latency_ms=latency_ms,
            created_at=utc_now_iso(),
        )
        if persist:
            self.store.save_job(result)
        return result

    def process_bytes(
        self,
        content: bytes,
        filename: str,
        content_type: str | None = None,
        doc_type: DocTypeRequest = "auto",
        provider_override: str | None = None,
        persist: bool = True,
    ) -> ExtractionResult:
        """Ingest raw file bytes (PDF, image, or text), extract, validate, and optionally store."""
        t0 = time.perf_counter()
        provider = (provider_override or self.settings.provider).lower()
        render_images = provider in {"ollama", "gemini", "openai"}
        doc = ingest_bytes(content, filename=filename, content_type=content_type, render_images=render_images)
        return self._run_on_ingested(
            doc,
            doc_type=doc_type,
            provider_override=provider_override,
            persist=persist,
            started_at=t0,
        )

    def process_text(
        self,
        text: str,
        filename: str = "document.txt",
        doc_type: DocTypeRequest = "auto",
        provider_override: str | None = None,
        persist: bool = True,
    ) -> ExtractionResult:
        """Ingest plain text, extract, validate, and optionally store."""
        t0 = time.perf_counter()
        doc = ingest_text(text, filename=filename)
        return self._run_on_ingested(
            doc,
            doc_type=doc_type,
            provider_override=provider_override,
            persist=persist,
            started_at=t0,
        )

    def process_file(
        self,
        path: Path,
        doc_type: DocTypeRequest = "auto",
        provider_override: str | None = None,
        persist: bool = True,
    ) -> ExtractionResult:
        """Ingest a file from disk, extract, validate, and optionally store."""
        t0 = time.perf_counter()
        provider = (provider_override or self.settings.provider).lower()
        render_images = provider in {"ollama", "gemini", "openai"}
        doc = ingest_file(path, render_images=render_images)
        return self._run_on_ingested(
            doc,
            doc_type=doc_type,
            provider_override=provider_override,
            persist=persist,
            started_at=t0,
        )

    def submit_review(self, job_id: str, submission: ReviewSubmission) -> ExtractionResult:
        """Apply human corrections, re-run arithmetic validation, and update HITL job status."""
        existing = self.store.get_job(job_id)
        if existing is None:
            raise KeyError(f"Extraction job '{job_id}' not found.")

        schema_cls = schema_for_doc_type(existing.doc_type)
        # Merge existing fields with submitted patch so partial or full edits both work cleanly
        merged_payload: dict[str, Any] = existing.data.model_dump()
        merged_payload.update(submission.data)
        updated_data = schema_cls.model_validate(merged_payload)

        # Mark human-submitted/verified fields with provenance='human'
        prev_sources = {fc.field_name: fc.source for fc in existing.field_confidences}
        for key in submission.data:
            prev_sources[key] = "human"
        if submission.approve:
            for key in schema_cls.model_fields:
                if getattr(updated_data, key) not in (None, "", []):
                    prev_sources[key] = "human"

        validation = validate_extraction(
            doc_type=existing.doc_type,
            data=updated_data,
            field_sources=prev_sources,
            review_confidence_threshold=self.settings.review_confidence_threshold,
        )

        # If reviewer explicitly approves AND reconciliation passes (or overrides with notes), mark REVIEWED
        if submission.approve and (validation.reconciliation_passed or submission.reviewer_notes):
            new_status = "REVIEWED"
        else:
            new_status = "NEEDS_REVIEW"

        updated_result = existing.model_copy(
            update={
                "status": new_status,
                "overall_confidence": validation.overall_confidence,
                "reconciliation_passed": validation.reconciliation_passed,
                "reconciliation_notes": validation.reconciliation_notes,
                "data": updated_data,
                "field_confidences": validation.field_confidences,
                "updated_at": utc_now_iso(),
                "reviewer_notes": submission.reviewer_notes,
            }
        )
        return self.store.update_job(updated_result)
