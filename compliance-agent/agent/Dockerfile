FROM python:3.11-slim AS agent-base

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

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -e /workspace/rag-core \
    && python -m pip install --no-cache-dir \
       -e "/workspace/compliance-agent/agent[dev]"

WORKDIR /workspace/compliance-agent/agent


FROM agent-base AS agent-test

COPY compliance-agent/agent/tests \
    /workspace/compliance-agent/agent/tests

CMD ["python", "-m", "pytest", "-q"]
