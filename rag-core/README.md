# Generic RAG Core

`generic-rag-core` is a provider-neutral Python package for traceable document
ingestion, chunking, immutable vector indexing, exact-scope retrieval, and
grounded answer generation.

The package contains no FastAPI, database, job queue, knowledge-base
membership, compliance-agent, or frontend runtime. Applications resolve
authorization and source membership before calling the core.

## Supported Phase 1 scope

- UTF-8 `.txt` documents.
- Text-based, single-column PDF documents through the optional PDF extra.
- Exact SHA-256 source identity.
- Block, fixed-window, and structure-aware chunking.
- Provider-neutral indexing and retrieval ports.
- Optional OpenAI embedding/generation adapters.
- Optional Qdrant indexing/search adapters.
- Citation labels linked to exact retrieved evidence.

Explicitly unsupported in the Phase 1 gate:

- OCR for scanned PDFs.
- Ambiguous multi-column PDF reading order.
- DOCX/Docling runtime.
- Sparse, hybrid, fusion, and reranking strategies.
- Knowledge-base membership and authorization decisions.
- Legal or compliance verdicts.

## Installation

Core only:

```bash
python -m pip install -e ./rag-core
```

Development and format/provider extras:

```bash
python -m pip install -e './rag-core[dev,pdf,openai,qdrant]'
```

The core dependency set contains only Pydantic. Provider SDKs remain optional
and are imported only from `rag_core.adapters`.

## Architecture

```text
local file
  -> ingestion: validate, checksum, extract, parse
  -> chunking: preserve source spans and deterministic IDs
  -> indexing: validate snapshot, embed, persist
  -> retrieval: exact scope, rank, isolate
  -> generation: render evidence, generate, validate citation labels
```

Provider-independent modules never construct OpenAI or Qdrant clients. The
consumer owns configuration, credentials, client lifetime, authorization, and
knowledge-base membership.

## Generic TXT example

```python
from pathlib import Path

from rag_core.chunking import StructureAwareChunkingStrategy
from rag_core.ingestion.extractors.text import TextDocumentExtractor
from rag_core.ingestion.models import DocumentInput, DocumentRole
from rag_core.ingestion.parsing import FlatStructureParser
from rag_core.ingestion.service import DocumentProcessor

processor = DocumentProcessor(
    extractors=(TextDocumentExtractor(),),
    parser=FlatStructureParser(),
    chunking_strategy=StructureAwareChunkingStrategy(
        max_characters=800,
        fallback_overlap_characters=80,
    ),
)

result = processor.process(
    DocumentInput(
        path=Path("refund-policy.txt"),
        source_id="refund-policy",
        source_version="2026-01",
        access_scope="tenant-a",
        role=DocumentRole.KNOWLEDGE,
    )
)

for chunk in result.chunks:
    print(chunk.chunk_id, chunk.structure_path, chunk.source_spans)
```

No page number or bounding box is invented for TXT input. Character offsets
must map back to the original file content.

## Composition with providers

Create provider clients in the consuming application, then inject adapters into
core services:

```python
from openai import OpenAI
from qdrant_client import QdrantClient

from rag_core.adapters.openai_embeddings import OpenAIEmbeddingProvider
from rag_core.adapters.qdrant import QdrantVectorIndex
from rag_core.indexing import DocumentIndexingService

dimensions = 1536
embedding_provider = OpenAIEmbeddingProvider(
    client=OpenAI(),
    model="text-embedding-3-small",
    dimensions=dimensions,
)
index_profile = (
    f"{embedding_provider.index_profile}:"
    "chunk=structure_aware:v1"
)
vector_index = QdrantVectorIndex(
    client=QdrantClient(url="http://qdrant:6333"),
    collection_name="knowledge",
    vector_dimensions=dimensions,
    index_profile=index_profile,
)
indexing_service = DocumentIndexingService(
    embedding_provider=embedding_provider,
    vector_index=vector_index,
    expected_dimensions=dimensions,
)
```

Never hardcode API keys in source files or copy `.env` files between projects.

## Snapshot identity

Retrieval and indexing isolate evidence with:

```text
access_scope
source_id
source_version
source_checksum_sha256
index_profile
chunk_id
```

The `index_profile` must identify the embedding model, vector dimension, and
chunking strategy/version. Changing any of these creates a different snapshot.
Knowledge-base IDs are not stored in vector payloads; backend persistence owns
membership.

## Failure behavior

Expected ingestion failures use stable typed codes such as:

- `unsupported_document_format`
- `invalid_file_signature`
- `invalid_text_encoding`
- `ocr_required`
- `layout_unsupported`
- `checksum_mismatch`

Indexing and retrieval fail explicitly for mixed snapshots, duplicate chunk
IDs, invalid vector dimensions, non-finite values, stale checksums, unknown
payload schemas, or evidence outside the requested scope.

Generation behavior:

- No retrieved evidence produces an `abstained` result without a model call.
- Grounded answers require at least one known citation label.
- Unknown or missing labels are rejected.
- Citation-label validation proves linkage, not semantic or legal correctness.

## Development checks

With Docker Compose:

```bash
docker compose --profile test build core-test
docker compose --profile test run --rm core-test ruff check src tests
docker compose --profile test run --rm core-test python -m pytest -q
docker compose --profile test run --rm core-test \
  python /workspace/scripts/check_boundaries.py
```

These commands provide offline structural and behavioral evidence. They do not
prove live OpenAI/Qdrant operation, retrieval quality, legal correctness,
security hardening, or production readiness.
