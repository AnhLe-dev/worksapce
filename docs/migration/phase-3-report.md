# Phase 3 report — Bounded compliance agent

## Scope

Phase 3 connects a generic processed contract to explicitly selected,
backend-authorized knowledge through a bounded agent. It does not migrate the
frontend and does not claim live-model or legal acceptance.

## Implemented flow

```text
POST review-jobs
  -> PostgreSQL technical job + transactional outbox
  -> Celery message containing job_id only
  -> worker reloads contract revision, review units and selected KB
  -> PostgreSQL resolves READY source snapshots and trusted scope
  -> bounded tools retrieve evidence and complete reference units
  -> deterministic artifact/lineage validator
  -> domain ReviewResult stored only when the technical job succeeds
```

## Decisions and reasons

1. The API requires an explicit `knowledge_base_id` because the model must
   never invent its legal authority.
2. Scope, snapshot IDs, checksums and evidence are not accepted from HTTP
   because these values are authorization and lineage claims owned by the
   backend.
3. Zero retrieval hits produce `INSUFFICIENT_EVIDENCE`; they do not prove that
   the selected knowledge contains no relevant rule.
4. Fabricated quotes or citations produce `VALIDATION_FAILED`, while provider
   crashes produce technical `job.status=failed`. This preserves the boundary
   between business outcomes and infrastructure failures.
5. Fake providers are used for the offline gate so the workflow is
   deterministic and free of credentials/cost. They do not prove live model
   quality.

## Batch 14 acceptance cases

- Applicable generic KB + exact evidence -> `REVIEW_COMPLETED` with reference
  lineage.
- Applicable KB + zero hits -> `INSUFFICIENT_EVIDENCE`.
- Fabricated contract quote -> `VALIDATION_FAILED` with structured issue.
- Ambiguous evidence -> `NEEDS_HUMAN_REVIEW`.
- Explicit policy mismatch -> `NO_APPLICABLE_KNOWLEDGE` without model call.
- Missing applicability proof -> `APPLICABILITY_UNDETERMINED`.
- Model crash -> technical `FAILED`, no domain result, contract remains READY.
- Cross-scope retriever output -> fail closed.
- Duplicate delivery -> handler is not executed twice.
- Alembic Phase 3 revision is the single migration head.

## Verification status

`OFFLINE_VERIFIED` from the workspace Compose test profile. The complete
Phase 3 gate was run from the workspace root:

```bash
cd /Users/matthew/Desktop/workspace
make phase3-verify
```

The command builds the agent/backend test images, validates Compose, runs Ruff
for both projects, then runs the complete agent and backend suites once.

If a previous run already built both images and stopped before pytest, resume
without rebuilding the images:

```bash
make phase3-check
```

Recorded result:

- Compose configuration: PASS.
- Agent Ruff: PASS.
- Agent pytest: PASS.
- Backend Ruff: PASS.
- Backend pytest: PASS — `58 passed, 1 warning in 4.27s`.
- Warning: upstream `StarletteDeprecationWarning` from
  `fastapi.testclient`; it does not fail the current offline gate.

## Acceptance boundary

This result means `OFFLINE_VERIFIED` for Phase 3. It does not mean
`LIVE_ACCEPTED`, production security accepted, semantic correctness accepted,
or legal correctness accepted. Live provider acceptance remains a separately
authorized gate.
