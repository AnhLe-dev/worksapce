"""Check rag-core dependency boundaries using Python's AST."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path

_PROJECT_IMPORT_PREFIXES = (
    "app",
    "backend",
    "compliance_agent",
    "rag_research",
)
_FORBIDDEN_RUNTIME_PREFIXES = (
    "celery",
    "fastapi",
    "langchain",
    "langchain_openai",
    "minio",
    "ragas",
    "sqlalchemy",
)
_PROVIDER_PREFIXES = (
    "openai",
    "qdrant_client",
)
_DOMAIN_MARKERS = (
    "mt700",
    "ucp600",
)
_ALLOWED_INTERNAL_LAYERS = {
    "adapters": {
        "adapters",
        "generation",
        "indexing",
        "ingestion",
        "retrieval",
    },
    "chunking": {"chunking", "ingestion"},
    "generation": {"generation", "ingestion", "retrieval"},
    "indexing": {"indexing"},
    "ingestion": {"chunking", "ingestion"},
    "retrieval": {"ingestion", "retrieval"},
}


@dataclass(frozen=True, order=True)
class Violation:
    """Describe one deterministic boundary violation."""

    path: str
    line: int
    message: str


def _matches_prefix(module_name: str, prefixes: tuple[str, ...]) -> bool:
    """Return whether an import equals or descends from one prefix."""

    return any(
        module_name == prefix
        or module_name.startswith(f"{prefix}.")
        for prefix in prefixes
    )


def _module_imports(tree: ast.AST) -> tuple[tuple[str, int], ...]:
    """Extract absolute import targets and source lines."""

    imports: list[tuple[str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                (alias.name, node.lineno)
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                imports.append((node.module, node.lineno))

    return tuple(imports)


def _domain_marker_violations(
    *,
    tree: ast.AST,
    relative_path: Path,
) -> tuple[Violation, ...]:
    """Find domain defaults in identifiers and executable strings."""

    violations: list[Violation] = []

    for node in ast.walk(tree):
        candidate: str | None = None

        if isinstance(node, ast.Name):
            candidate = node.id
        elif isinstance(node, ast.Attribute):
            candidate = node.attr
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            candidate = node.value

        if candidate is None:
            continue

        normalized = candidate.casefold()
        matched = next(
            (
                marker
                for marker in _DOMAIN_MARKERS
                if marker in normalized
            ),
            None,
        )

        if matched is not None:
            violations.append(
                Violation(
                    path=str(relative_path),
                    line=getattr(node, "lineno", 1),
                    message=(
                        "generic runtime contains domain marker "
                        f"{matched!r}"
                    ),
                )
            )

    return tuple(violations)


def check_boundaries(core_root: Path) -> tuple[Violation, ...]:
    """Return every import, layer, and domain boundary violation."""

    violations: list[Violation] = []

    for source_path in sorted(core_root.rglob("*.py")):
        relative_path = source_path.relative_to(core_root)

        try:
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                filename=str(source_path),
            )
        except SyntaxError as exc:
            violations.append(
                Violation(
                    path=str(relative_path),
                    line=exc.lineno or 1,
                    message=f"syntax error: {exc.msg}",
                )
            )
            continue

        layer = relative_path.parts[0]

        for module_name, line_number in _module_imports(tree):
            if _matches_prefix(
                module_name,
                _PROJECT_IMPORT_PREFIXES,
            ):
                violations.append(
                    Violation(
                        path=str(relative_path),
                        line=line_number,
                        message=(
                            "imports application-owned module "
                            f"{module_name!r}"
                        ),
                    )
                )

            if _matches_prefix(
                module_name,
                _FORBIDDEN_RUNTIME_PREFIXES,
            ):
                violations.append(
                    Violation(
                        path=str(relative_path),
                        line=line_number,
                        message=(
                            "imports forbidden runtime dependency "
                            f"{module_name!r}"
                        ),
                    )
                )

            if (
                _matches_prefix(module_name, _PROVIDER_PREFIXES)
                and layer != "adapters"
            ):
                violations.append(
                    Violation(
                        path=str(relative_path),
                        line=line_number,
                        message=(
                            "provider SDK import is only allowed in "
                            f"adapters: {module_name!r}"
                        ),
                    )
                )

            if module_name == "pymupdf" and relative_path != Path(
                "ingestion/extractors/pdf.py"
            ):
                violations.append(
                    Violation(
                        path=str(relative_path),
                        line=line_number,
                        message=(
                            "PyMuPDF import is only allowed in the "
                            "PDF extractor adapter"
                        ),
                    )
                )

            if not module_name.startswith("rag_core."):
                continue

            imported_parts = module_name.split(".")

            if len(imported_parts) < 2:
                continue

            imported_layer = imported_parts[1]
            allowed_layers = _ALLOWED_INTERNAL_LAYERS.get(layer)

            if (
                allowed_layers is not None
                and imported_layer not in allowed_layers
            ):
                violations.append(
                    Violation(
                        path=str(relative_path),
                        line=line_number,
                        message=(
                            f"layer {layer!r} cannot import "
                            f"rag_core.{imported_layer}"
                        ),
                    )
                )

        violations.extend(
            _domain_marker_violations(
                tree=tree,
                relative_path=relative_path,
            )
        )

    return tuple(sorted(set(violations)))


def main() -> int:
    """Run the boundary audit and return a shell-friendly status."""

    default_root = (
        Path(__file__).resolve().parents[1]
        / "rag-core"
        / "src"
        / "rag_core"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core-root",
        type=Path,
        default=default_root,
    )
    arguments = parser.parse_args()
    core_root = arguments.core_root.resolve()

    if not core_root.is_dir():
        parser.error(
            f"rag-core source directory does not exist: {core_root}"
        )

    violations = check_boundaries(core_root)

    if violations:
        for violation in violations:
            print(
                f"{violation.path}:{violation.line}: "
                f"{violation.message}"
            )

        print(f"Boundary check failed: {len(violations)} violation(s)")
        return 1

    source_count = len(tuple(core_root.rglob("*.py")))
    print(
        "Boundary check passed: "
        f"{source_count} Python source file(s) inspected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
