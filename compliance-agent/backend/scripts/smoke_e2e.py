"""Small public-surface smoke probe without provider acceptance claims."""

from argparse import ArgumentParser

import httpx


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=10) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")
        schema = client.get("/openapi.json")

    for response in (live, ready, schema):
        response.raise_for_status()

    paths = schema.json()["paths"]
    required = {
        "/api/v1/contracts",
        "/api/v1/knowledge-bases",
        "/api/v1/jobs/{job_id}",
    }

    if not required.issubset(paths):
        raise RuntimeError("Release A OpenAPI surface is incomplete")

    print("Release A API is live, ready, and structurally complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
