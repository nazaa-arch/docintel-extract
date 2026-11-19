"""Configuration management via DOCINTEL_* environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ProviderType = Literal["local", "ollama", "gemini", "openai"]


def _load_dotenv_file(dotenv_path: Path) -> dict[str, str]:
    """Parse a simple .env file without requiring external packages."""
    parsed: dict[str, str] = {}
    if not dotenv_path.is_file():
        return parsed
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            parsed[key] = value
    return parsed


def _env_value(key: str, dotenv_values: dict[str, str], default: str | None = None) -> str | None:
    if key in os.environ:
        return os.environ[key]
    if key in dotenv_values:
        return dotenv_values[key]
    return default


class Settings(BaseModel):
    """Runtime settings for DocIntel Extract."""

    provider: ProviderType = Field(default="local")
    model: str = Field(default="qwen2.5vl:7b")
    ollama_base_url: str = Field(default="http://localhost:11434")
    gemini_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)
    openai_base_url: str = Field(default="https://api.openai.com/v1")
    review_confidence_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    db_path: Path = Field(default=Path("data/docintel_review.db"))
    upload_dir: Path = Field(default=Path("data/uploads"))
    samples_dir: Path = Field(default=Path("data/samples"))
    http_timeout_seconds: float = Field(default=45.0, gt=0.0)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)

    @classmethod
    def from_env(cls, dotenv_path: Path | None = None) -> Settings:
        """Load settings from DOCINTEL_* environment variables and optional .env file."""
        env_file = dotenv_path or Path(".env")
        dotenv_values = _load_dotenv_file(env_file)

        provider = (_env_value("DOCINTEL_PROVIDER", dotenv_values, "local") or "local").lower()
        if provider not in {"local", "ollama", "gemini", "openai"}:
            provider = "local"

        threshold_raw = _env_value("DOCINTEL_REVIEW_CONFIDENCE_THRESHOLD", dotenv_values, "0.80")
        timeout_raw = _env_value("DOCINTEL_HTTP_TIMEOUT_SECONDS", dotenv_values, "45.0")
        port_raw = _env_value("DOCINTEL_PORT", dotenv_values, "8000")

        return cls(
            provider=provider,  # type: ignore[arg-type]
            model=_env_value("DOCINTEL_MODEL", dotenv_values, "qwen2.5vl:7b") or "qwen2.5vl:7b",
            ollama_base_url=(
                _env_value("DOCINTEL_OLLAMA_BASE_URL", dotenv_values, "http://localhost:11434")
                or "http://localhost:11434"
            ).rstrip("/"),
            gemini_api_key=_env_value("DOCINTEL_GEMINI_API_KEY", dotenv_values) or None,
            openai_api_key=_env_value("DOCINTEL_OPENAI_API_KEY", dotenv_values) or None,
            openai_base_url=(
                _env_value("DOCINTEL_OPENAI_BASE_URL", dotenv_values, "https://api.openai.com/v1")
                or "https://api.openai.com/v1"
            ).rstrip("/"),
            review_confidence_threshold=float(threshold_raw or "0.80"),
            db_path=Path(
                _env_value("DOCINTEL_DB_PATH", dotenv_values, "data/docintel_review.db")
                or "data/docintel_review.db"
            ),
            upload_dir=Path(
                _env_value("DOCINTEL_UPLOAD_DIR", dotenv_values, "data/uploads") or "data/uploads"
            ),
            samples_dir=Path(
                _env_value("DOCINTEL_SAMPLES_DIR", dotenv_values, "data/samples")
                or "data/samples"
            ),
            http_timeout_seconds=float(timeout_raw or "45.0"),
            host=_env_value("DOCINTEL_HOST", dotenv_values, "0.0.0.0") or "0.0.0.0",
            port=int(port_raw or "8000"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached runtime settings."""
    return Settings.from_env()
