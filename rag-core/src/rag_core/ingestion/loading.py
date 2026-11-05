"""Validate local document inputs and create their source identity."""

from hashlib import sha256
from pathlib import Path

from rag_core.ingestion.errors import (
    DocumentChecksumMismatchError,
    EmptyDocumentError,
    InvalidFileSignatureError,
    UnsupportedDocumentFormatError,
)
from rag_core.ingestion.models import (
    DocumentFormat,
    DocumentInput,
    DocumentSource,
)

_READ_BUFFER_SIZE = 1024 * 1024
_PDF_SIGNATURE = b"%PDF-"


def validate_document_path(file_path: Path) -> Path:
    """Validate requirements shared by every local document."""

    resolved_path = file_path.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Source document does not exist: {resolved_path}"
        )

    if not resolved_path.is_file():
        raise ValueError(
            f"Source document is not a file: {resolved_path}"
        )

    if resolved_path.stat().st_size == 0:
        raise EmptyDocumentError(
            f"Source document is empty: {resolved_path}"
        )

    return resolved_path


def validate_pdf_path(file_path: Path) -> Path:
    """Validate a PDF extension and signature for compatibility callers."""

    resolved_path = validate_document_path(file_path)

    if resolved_path.suffix.casefold() != ".pdf":
        raise ValueError(
            f"Source document must have a .pdf extension: {resolved_path}"
        )

    with resolved_path.open("rb") as source_file:
        signature = source_file.read(len(_PDF_SIGNATURE))

    if signature != _PDF_SIGNATURE:
        raise InvalidFileSignatureError(
            "Source document does not have a valid PDF signature: "
            f"{resolved_path}"
        )

    return resolved_path


def detect_document_format(file_path: Path) -> DocumentFormat:
    """Detect supported formats without guessing from content alone."""

    resolved_path = validate_document_path(file_path)
    suffix = resolved_path.suffix.casefold()

    if suffix == ".pdf":
        validate_pdf_path(resolved_path)
        return DocumentFormat.PDF

    if suffix == ".txt":
        with resolved_path.open("rb") as source_file:
            signature = source_file.read(len(_PDF_SIGNATURE))

        if signature == _PDF_SIGNATURE:
            raise InvalidFileSignatureError(
                "Document extension does not match its PDF signature"
            )

        return DocumentFormat.TXT

    raise UnsupportedDocumentFormatError(
        f"Unsupported document format: {suffix or '<none>'}"
    )


def calculate_file_sha256(file_path: Path) -> str:
    """Calculate SHA-256 without loading the complete file into memory."""

    resolved_path = validate_document_path(file_path)
    digest = sha256()

    with resolved_path.open("rb") as source_file:
        while file_block := source_file.read(_READ_BUFFER_SIZE):
            digest.update(file_block)

    return digest.hexdigest()


def load_document_source(
    document_input: DocumentInput,
) -> DocumentSource:
    """Validate one input and create its immutable source descriptor."""

    resolved_path = validate_document_path(
        document_input.path
    )
    document_format = detect_document_format(
        resolved_path
    )

    return DocumentSource(
        source_id=document_input.source_id,
        filename=resolved_path.name,
        source_version=document_input.source_version,
        checksum_sha256=calculate_file_sha256(
            resolved_path
        ),
        access_scope=document_input.access_scope,
        role=document_input.role,
        document_format=document_format,
        metadata=dict(document_input.metadata),
    )


def validate_source_file(
    source: DocumentSource,
    file_path: Path,
) -> Path:
    """Require a path to match an already-created source descriptor."""

    resolved_path = validate_document_path(file_path)
    actual_format = detect_document_format(
        resolved_path
    )

    if actual_format is not source.document_format:
        raise ValueError(
            "Document format does not match its source descriptor"
        )

    actual_checksum = calculate_file_sha256(
        resolved_path
    )

    if actual_checksum != source.checksum_sha256:
        raise DocumentChecksumMismatchError(
            "Document checksum does not match its source descriptor"
        )

    return resolved_path
