class RAGError(Exception):
    """Base class for application errors."""


class UnsupportedFileTypeError(RAGError):
    pass


class DocumentParseError(RAGError):
    """The file could not be read, or contained no extractable text."""


class FileTooLargeError(RAGError):
    pass


class TooManyChunksError(RAGError):
    """The document would produce more chunks than RAG_MAX_CHUNKS_PER_DOCUMENT."""


class UpstreamModelError(RAGError):
    """An embedding or LLM backend failed or returned an invalid response."""


class IndexCompatibilityError(RAGError):
    """The persisted index was built with a different embedding model."""


class WorkspaceNotFoundError(RAGError):
    pass


class WorkspaceExistsError(RAGError):
    pass


class InvalidWorkspaceError(RAGError):
    """Bad workspace name, or an operation the workspace doesn't allow (deleting the default one)."""
