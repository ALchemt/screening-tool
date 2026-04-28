# HF Spaces Docker SDK target. Listens on $PORT (default 7860).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

# HF Spaces writes to /tmp; persist sqlite + chroma there.
ENV CHECKPOINT_DB=/tmp/checkpoints.sqlite \
    CHROMA_DIR=/tmp/chroma \
    PORT=7860

# Build ChromaDB index at container start (idempotent — uses /tmp/chroma).
# OPENROUTER_API_KEY must be set in HF Space secrets.
CMD ["sh", "-c", "python scripts/build_index.py && uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]

EXPOSE 7860
