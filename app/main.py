from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.api.routes import UPSTREAM_ERROR_MESSAGE, router
from app.config import get_settings
from app.errors import (
    DocumentParseError,
    FileTooLargeError,
    IndexCompatibilityError,
    InvalidWorkspaceError,
    TooManyChunksError,
    UnsupportedFileTypeError,
    UpstreamModelError,
    WorkspaceExistsError,
    WorkspaceNotFoundError,
)
from app.observability import ObservabilityMiddleware, configure_logging
from app.service import RAGService

logger = logging.getLogger("app")

_ERROR_STATUS = {
    UnsupportedFileTypeError: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    FileTooLargeError: status.HTTP_413_CONTENT_TOO_LARGE,
    TooManyChunksError: status.HTTP_413_CONTENT_TOO_LARGE,
    DocumentParseError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    IndexCompatibilityError: status.HTTP_409_CONFLICT,
    UpstreamModelError: status.HTTP_503_SERVICE_UNAVAILABLE,
    WorkspaceNotFoundError: status.HTTP_404_NOT_FOUND,
    WorkspaceExistsError: status.HTTP_409_CONFLICT,
    InvalidWorkspaceError: status.HTTP_400_BAD_REQUEST,
}


def create_app(service: RAGService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "service", None) is None:
            app.state.service = RAGService(get_settings())
            if app.state.service.settings.warmup:
                threading.Thread(target=app.state.service.warm_up, daemon=True).start()
        app.state.service.start()
        yield
        app.state.service.stop()

    app = FastAPI(
        title="Grounded RAG API",
        version="1.1.0",
        description="Ingest PDF/Word/Excel/CSV/TXT/Markdown documents into workspaces and ask questions answered with cited "
        "evidence.",
        lifespan=lifespan,
    )
    app.state.service = service
    app.add_middleware(ObservabilityMiddleware)
    app.include_router(router)

    for error_type, code in _ERROR_STATUS.items():
        app.add_exception_handler(error_type, _handler(code))
    return app


def _handler(code: int):
    async def handle(request: Request, exc: Exception) -> JSONResponse:
        if code >= 500:
            logger.warning("Upstream failure on %s: %s", request.url.path, exc)
            detail = UPSTREAM_ERROR_MESSAGE
        else:
            detail = str(exc)
        return JSONResponse(status_code=code, content={"detail": detail})

    return handle


configure_logging(get_settings().log_format, get_settings().log_level)
app = create_app()
