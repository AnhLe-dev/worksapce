"""Fail fast when API-critical local dependencies are unavailable."""

import json

from app.api.runtime import create_api_runtime
from app.core.config import get_settings


def main() -> int:
    runtime = create_api_runtime(get_settings())

    try:
        checks = runtime.readiness_checks()
    finally:
        runtime.close()

    print(json.dumps(checks, sort_keys=True))
    return 0 if all(value == "ready" for value in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
