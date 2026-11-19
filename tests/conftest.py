"""Pytest fixtures for isolated DocIntel Extract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docintel.api import create_app
from docintel.config import Settings
from docintel.pipeline import DocumentIntelligencePipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = REPO_ROOT / "data" / "samples"


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    """Isolated settings backed by a temporary SQLite database."""
    return Settings(
        provider="local",
        db_path=tmp_path / "test_docintel.db",
        upload_dir=tmp_path / "uploads",
        samples_dir=SAMPLES_DIR,
        review_confidence_threshold=0.80,
    )


@pytest.fixture()
def pipeline(test_settings: Settings) -> DocumentIntelligencePipeline:
    return DocumentIntelligencePipeline(test_settings)


@pytest.fixture()
def client(test_settings: Settings) -> TestClient:
    app = create_app(test_settings)
    with TestClient(app) as test_client:
        yield test_client
