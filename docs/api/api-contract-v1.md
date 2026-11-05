# API contract v1 — Release A + Phase 3 review jobs

Release A exposes generic document, knowledge-base, and technical-job
lifecycle operations. It does not claim domain review or grounded-answer
acceptance. Phase 3 adds one Release B endpoint that queues bounded review;
frontend/source-viewer compatibility remains deferred to Phase 4.

## Trusted boundary

- `access_scope` comes only from `SYSTEM_ACCESS_SCOPE`.
- Contract and knowledge roles are selected by the route.
- `index_profile` is derived from the server embedding/chunking settings.
- Clients may supply source identity, source version, expected processing
  revision, and idempotency keys; they cannot override isolation metadata.
- Review clients must explicitly select a `knowledge_base_id` and current
  review-unit IDs. The backend resolves scope, memberships, snapshots and
  source lineage again inside the worker.

## Endpoints

| Method | Path | Success | Purpose |
|---|---|---:|---|
| POST | `/api/v1/contracts` | 201 | Store and register a contract |
| GET | `/api/v1/contracts/{id}` | 200 | Read contract lifecycle state |
| POST | `/api/v1/contracts/{id}/processing-jobs` | 202 | Queue generic processing |
| GET | `/api/v1/contracts/{id}/review-units` | 200 | Read current provenance units |
| POST | `/api/v1/contracts/{id}/review-jobs` | 202 | Queue bounded review against one selected KB |
| POST | `/api/v1/knowledge-bases` | 201 | Create a knowledge base |
| GET | `/api/v1/knowledge-bases` | 200 | List scoped knowledge bases |
| POST | `/api/v1/knowledge-bases/{id}/documents` | 202 | Register source and queue indexing |
| GET | `/api/v1/knowledge-bases/{id}` | 200 | Read base and immutable snapshots |
| GET | `/api/v1/jobs/{id}` | 200 | Read technical job state |
| GET | `/health/live` | 200 | Confirm process liveness |
| GET | `/health/ready` | 200/503 | Confirm database and storage readiness |

Uploads are multipart `.pdf` or `.txt` files. JSON validation errors use
FastAPI's `422` response. Domain/API failures return:

```json
{
  "code": "stable_machine_code",
  "message": "Safe explanation"
}
```

Important failure conditions are `404` for scoped absence, `409` for an
identity conflict, `413` for the configured size limit, `415` for an
unsupported extension, and `422` for invalid or empty document bytes.

## Review job request

```json
{
  "knowledge_base_id": "<uuid>",
  "expected_processing_revision": 1,
  "review_unit_ids": ["<uuid>"],
  "question": "Optional bounded review question",
  "idempotency_key": "client-stable-key"
}
```

`access_scope`, source snapshot IDs, checksums, index profile and evidence are
intentionally absent. A successful technical job stores one of the seven
domain outcomes in `result_payload.outcome`; provider/DB/worker errors use
`job.status=failed` and do not impersonate a domain outcome.

The seven outcomes are `NO_KNOWLEDGE_BASE`,
`NO_APPLICABLE_KNOWLEDGE`, `APPLICABILITY_UNDETERMINED`,
`INSUFFICIENT_EVIDENCE`, `REVIEW_COMPLETED`, `VALIDATION_FAILED`, and
`NEEDS_HUMAN_REVIEW`.

## Deferred endpoints

Generic RAG question answering, contract source streaming and frontend
compatibility routes remain deferred. Phase 3 review is offline-verifiable
with injected fake providers; that evidence does not establish live-model,
semantic, legal, security or production acceptance.
