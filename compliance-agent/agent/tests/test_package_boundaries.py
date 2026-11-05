"""Keep the agent independent from backend and legacy domain runtimes."""

import ast
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "compliance_agent"
FORBIDDEN_IMPORTS = (
    "app",
    "backend",
    "fastapi",
    "minio",
    "qdrant_client",
    "sqlalchemy",
)
LEGACY_MARKERS = ("ucp600", "mt700")


def iter_imports(tree: ast.AST) -> tuple[str, ...]:
    imports: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    return tuple(imports)


def test_agent_has_no_backend_or_data_provider_imports() -> None:
    violations: list[str] = []

    for source_path in sorted(SOURCE_ROOT.glob("*.py")):
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))

        for module_name in iter_imports(tree):
            if any(
                module_name == prefix
                or module_name.startswith(f"{prefix}.")
                for prefix in FORBIDDEN_IMPORTS
            ):
                violations.append(f"{source_path.name}: {module_name}")

        normalized_source = source.casefold()
        for marker in LEGACY_MARKERS:
            if marker in normalized_source:
                violations.append(f"{source_path.name}: {marker}")

    assert violations == []
