FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

COPY scripts/ ./scripts/
COPY eval/ ./eval/
COPY data/samples/ ./data/samples/

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/state/uploads /app/data/uploads \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "docintel.api:app", "--host", "0.0.0.0", "--port", "8000"]
