# Phase 1 Report — Independent Generic RAG Core

```text
PHASE: P1 - RAG CORE PACKAGE HOÀN TOÀN ĐỘC LẬP
BASELINE COMMIT: 721cf2e0d027a630acf9ce530f38df9fceaa1a1f
BASELINE WORKTREE: DIRTY; pre-existing user changes preserved
LIVE PROVIDERS USED: no
GATE: NOT_RUN after final exit-gate changes; do not treat as PASS
NEXT: run the four verification commands in this report
```

## Changed files

- `rag-core/pyproject.toml`
- `rag-core/Dockerfile`
- `rag-core/README.md`
- `rag-core/src/rag_core/__init__.py`
- `rag-core/src/rag_core/adapters/openai_chat.py`
- `rag-core/src/rag_core/adapters/openai_embeddings.py`
- `rag-core/src/rag_core/adapters/qdrant.py`
- `rag-core/src/rag_core/chunking/block.py`
- `rag-core/src/rag_core/chunking/fixed.py`
- `rag-core/src/rag_core/chunking/ports.py`
- `rag-core/src/rag_core/chunking/structure_aware.py`
- `rag-core/src/rag_core/generation/citations.py`
- `rag-core/src/rag_core/generation/context.py`
- `rag-core/src/rag_core/generation/models.py`
- `rag-core/src/rag_core/generation/ports.py`
- `rag-core/src/rag_core/generation/prompt.py`
- `rag-core/src/rag_core/generation/service.py`
- `rag-core/src/rag_core/indexing/errors.py`
- `rag-core/src/rag_core/indexing/models.py`
- `rag-core/src/rag_core/indexing/payloads.py`
- `rag-core/src/rag_core/indexing/ports.py`
- `rag-core/src/rag_core/indexing/service.py`
- `rag-core/src/rag_core/ingestion/extractors/pdf.py`
- `rag-core/src/rag_core/ingestion/extractors/text.py`
- `rag-core/src/rag_core/ingestion/loading.py`
- `rag-core/src/rag_core/ingestion/models.py`
- `rag-core/src/rag_core/ingestion/parsing.py`
- `rag-core/src/rag_core/ingestion/ports.py`
- `rag-core/src/rag_core/ingestion/service.py`
- `rag-core/src/rag_core/retrieval/dense.py`
- `rag-core/src/rag_core/retrieval/models.py`
- `rag-core/src/rag_core/retrieval/ports.py`
- `rag-core/src/rag_core/retrieval/service.py`
- `scripts/check_boundaries.py`
- `compose.yaml`
- `Makefile`
- `.github/workflows/ci.yml`

## New files

- `rag-core/src/rag_core/adapters/__init__.py`
- `rag-core/src/rag_core/chunking/__init__.py`
- `rag-core/src/rag_core/generation/__init__.py`
- `rag-core/src/rag_core/generation/errors.py`
- `rag-core/src/rag_core/indexing/__init__.py`
- `rag-core/src/rag_core/ingestion/__init__.py`
- `rag-core/src/rag_core/ingestion/errors.py`
- `rag-core/src/rag_core/ingestion/extractors/__init__.py`
- `rag-core/src/rag_core/retrieval/__init__.py`
- `rag-core/src/rag_core/retrieval/errors.py`
- `rag-core/tests/unit/test_chunking_strategies.py`
- `rag-core/tests/unit/test_document_pipeline.py`
- `rag-core/tests/unit/test_generation_core.py`
- `rag-core/tests/unit/test_indexing_core.py`
- `rag-core/tests/unit/test_pdf_pipeline.py`
- `rag-core/tests/unit/test_provider_adapters.py`
- `rag-core/tests/unit/test_retrieval_core.py`
- `rag-core/tests/smoke/test_phase1_smoke.py`

## Removed from the new runtime

- No `.env`, credentials, raw legal corpus, UCP-specific runtime, MT700 parser,
  FastAPI, Celery, SQLAlchemy, MinIO, or Ragas dependencies were copied into
  `rag-core`.
- Legacy source files remain preserved in the old project; no deletion was
  performed.

## Design decisions

1. Core models and services depend on small Protocol ports; provider SDKs live
   only in `rag_core.adapters`.
2. Source identity includes scope, source, version, SHA-256 checksum, index
   profile, and chunk ID.
3. Knowledge-base membership remains backend/PostgreSQL-owned and is rejected
   from vector metadata.
4. Fixed chunking rejects unsafe PDF sub-block splitting when exact geometry is
   unavailable.
5. Retrieval fails closed when returned lineage differs from the requested
   immutable snapshot.
6. No evidence produces deterministic abstention without calling a model.
7. Citation validation proves label-to-evidence linkage only; it does not prove
   semantic entailment or legal correctness.

## Verification commands

```bash
docker compose --profile test build core-import core-test
```

```bash
docker compose --profile test run --rm core-import
```

```bash
docker compose --profile test run --rm core-test ruff check src tests
```

```bash
docker compose --profile test run --rm core-test python -m pytest -q
```

```bash
docker compose --profile test run --rm core-test \
  python /workspace/scripts/check_boundaries.py
```

Record exact counts and output here after execution. Earlier batch-level passes
do not establish the final gate after chunking, smoke, packaging, and boundary
changes.

## Known limitations

- Docling remains `NOT_IMPLEMENTED`.
- Sparse, hybrid, fusion, and reranking remain `NOT_IMPLEMENTED` for Release C.
- OCR and ambiguous multi-column PDF layouts are rejected explicitly.
- Tests are offline and do not establish live OpenAI/Qdrant acceptance.
- Citation mapping does not establish semantic or legal correctness.
- Production security, authorization, jobs, API, and persistence belong to
  later phases.
