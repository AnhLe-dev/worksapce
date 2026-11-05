"""Generic document ingestion.

Import concrete contracts from ``models``, orchestration from ``service``,
and optional format extractors from ``extractors``. Keeping this package
initializer lazy prevents chunking/ingestion import cycles and avoids loading
optional PDF dependencies during a core-only import.
"""
