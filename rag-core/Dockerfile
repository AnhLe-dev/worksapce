FROM python:3.11-slim AS core-import

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /workspace

COPY rag-core/pyproject.toml /workspace/rag-core/pyproject.toml
COPY rag-core/src /workspace/rag-core/src
COPY rag-core/README.md /workspace/rag-core/README.md

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir \
       -e "/workspace/rag-core"

WORKDIR /workspace/rag-core

CMD ["python", "-c", "import rag_core; print(rag_core.__version__)"]

FROM core-import AS test

COPY rag-core/tests /workspace/rag-core/tests
COPY scripts /workspace/scripts

RUN python -m pip install --no-cache-dir \
       -e "/workspace/rag-core[dev,openai,pdf,qdrant]"

WORKDIR /workspace/rag-core

CMD ["pytest", "-q"]
