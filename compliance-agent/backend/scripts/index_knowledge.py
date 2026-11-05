"""Register one local knowledge source through the public API."""

from argparse import ArgumentParser
from pathlib import Path

import httpx


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("knowledge_base_id")
    parser.add_argument("file", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-version", required=True)
    parser.add_argument("--idempotency-key", required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.file.resolve()

    with path.open("rb") as source:
        response = httpx.post(
            f"{args.base_url}/api/v1/knowledge-bases/"
            f"{args.knowledge_base_id}/documents",
            files={"file": (path.name, source)},
            data={
                "source_id": args.source_id,
                "source_version": args.source_version,
                "idempotency_key": args.idempotency_key,
            },
            timeout=60,
        )

    response.raise_for_status()
    print(response.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
