"""Typed failures raised by generic document ingestion."""

from enum import StrEnum


class DocumentErrorCode(StrEnum):
    """Stable error codes exposed to application consumers."""

    EMPTY_DOCUMENT = "empty_document"
    INVALID_FILE_SIGNATURE = "invalid_file_signature"
    UNSUPPORTED_DOCUMENT_FORMAT = "unsupported_document_format"
    INVALID_TEXT_ENCODING = "invalid_text_encoding"
    INVALID_PDF = "invalid_pdf"
    ENCRYPTED_DOCUMENT = "encrypted_document"
    OCR_REQUIRED = "ocr_required"
    LAYOUT_UNSUPPORTED = "layout_unsupported"
    CHECKSUM_MISMATCH = "checksum_mismatch"


class DocumentProcessingError(ValueError):
    """Base error containing a stable machine-readable code."""

    code: DocumentErrorCode

    def __init__(
        self,
        code: DocumentErrorCode,
        message: str,
    ) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code.value}: {message}")


class EmptyDocumentError(DocumentProcessingError):
    def __init__(self, message: str) -> None:
        super().__init__(
            DocumentErrorCode.EMPTY_DOCUMENT,
            message,
        )


class InvalidFileSignatureError(DocumentProcessingError):
    def __init__(self, message: str) -> None:
        super().__init__(
            DocumentErrorCode.INVALID_FILE_SIGNATURE,
            message,
        )


class UnsupportedDocumentFormatError(DocumentProcessingError):
    def __init__(self, message: str) -> None:
        super().__init__(
            DocumentErrorCode.UNSUPPORTED_DOCUMENT_FORMAT,
            message,
        )


class InvalidTextEncodingError(DocumentProcessingError):
    def __init__(self, message: str) -> None:
        super().__init__(
            DocumentErrorCode.INVALID_TEXT_ENCODING,
            message,
        )


class InvalidPdfError(DocumentProcessingError):
    def __init__(self, message: str) -> None:
        super().__init__(
            DocumentErrorCode.INVALID_PDF,
            message,
        )


class EncryptedDocumentError(DocumentProcessingError):
    def __init__(self, message: str) -> None:
        super().__init__(
            DocumentErrorCode.ENCRYPTED_DOCUMENT,
            message,
        )


class OcrRequiredError(DocumentProcessingError):
    def __init__(self, message: str) -> None:
        super().__init__(
            DocumentErrorCode.OCR_REQUIRED,
            message,
        )


class LayoutUnsupportedError(DocumentProcessingError):
    def __init__(self, message: str) -> None:
        super().__init__(
            DocumentErrorCode.LAYOUT_UNSUPPORTED,
            message,
        )


class DocumentChecksumMismatchError(DocumentProcessingError):
    def __init__(self, message: str) -> None:
        super().__init__(
            DocumentErrorCode.CHECKSUM_MISMATCH,
            message,
        )