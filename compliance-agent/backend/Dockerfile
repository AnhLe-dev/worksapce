FROM python:3.11-slim AS backend-base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

COPY rag-core/pyproject.toml /workspace/rag-core/pyproject.toml
COPY rag-core/src /workspace/rag-core/src
COPY rag-core/README.md /workspace/rag-core/README.md

COPY compliance-agent/agent/pyproject.toml \
    /workspace/compliance-agent/agent/pyproject.toml
COPY compliance-agent/agent/src \
    /workspace/compliance-agent/agent/src

COPY compliance-agent/backend/pyproject.toml \
    /workspace/compliance-agent/backend/pyproject.toml
COPY compliance-agent/backend/app \
    /workspace/compliance-agent/backend/app

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
       -e "/workspace/rag-core[pdf,openai,qdrant]" \
    && python -m pip install --no-cache-dir \
       -e "/workspace/compliance-agent/agent" \
    && python -m pip install --no-cache-dir \
       -e "/workspace/compliance-agent/backend[dev,postgres]"

WORKDIR /workspace/compliance-agent/backend


FROM backend-base AS backend-runtime

COPY compliance-agent/backend/alembic.ini \
    /workspace/compliance-agent/backend/alembic.ini
COPY compliance-agent/backend/migrations \
    /workspace/compliance-agent/backend/migrations
COPY compliance-agent/backend/scripts \
    /workspace/compliance-agent/backend/scripts

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


FROM backend-runtime AS backend-test

COPY compliance-agent/backend/tests \
    /workspace/compliance-agent/backend/tests

CMD ["python", "-m", "pytest", "-q"]
