.PHONY: core-build core-import core-lint core-test core-boundaries core-verify \
	agent-build agent-lint agent-test agent-verify \
	backend-build backend-lint backend-test backend-verify \
	phase3-check phase3-verify test-a test-b \
	live-build up down smoke

core-build:
	docker compose --profile test build core-import core-test

core-import:
	docker compose --profile test run --rm core-import

core-lint:
	docker compose --profile test run --rm core-test ruff check src tests

core-test:
	docker compose --profile test run --rm core-test python -m pytest -q

core-boundaries:
	docker compose --profile test run --rm core-test \
		python /workspace/scripts/check_boundaries.py

core-verify: core-import core-lint core-test core-boundaries

agent-build:
	docker compose --profile test build agent-test

agent-lint:
	docker compose --profile test run --rm agent-test ruff check src tests

agent-test:
	docker compose --profile test run --rm agent-test python -m pytest -q

agent-verify: agent-lint agent-test

backend-build:
	docker compose --profile test build backend-test

backend-lint:
	docker compose --profile test run --rm backend-test \
		ruff check app migrations scripts tests

backend-test:
	docker compose --profile test run --rm backend-test \
		python -m pytest -q

backend-verify: backend-lint backend-test

phase3-check:
	docker compose config -q
	docker compose --profile test run --rm agent-test \
		ruff check src tests
	docker compose --profile test run --rm agent-test \
		python -m pytest -q
	docker compose --profile test run --rm backend-test \
		ruff check app migrations scripts tests
	docker compose --profile test run --rm backend-test \
		python -m pytest -q

phase3-verify: agent-build backend-build
	$(MAKE) phase3-check

test-a: core-verify backend-verify

test-b: core-verify phase3-verify

live-build:
	docker compose build backend

up:
	docker compose up -d --build \
		postgres redis minio object-storage-init qdrant backend worker job-maintenance

down:
	docker compose down

smoke:
	docker compose exec -T backend python scripts/live_acceptance.py \
		--commit-sha "$$(git rev-parse --verify HEAD)"
