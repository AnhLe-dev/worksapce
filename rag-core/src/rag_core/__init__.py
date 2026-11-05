"""Provider-neutral RAG core package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("generic-rag-core")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = ("__version__",)
