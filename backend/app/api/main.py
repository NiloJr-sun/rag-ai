"""FastAPI application (T2.1).

Structure: this module owns the app and its wiring only. Endpoints that do
real work call into app.pipeline, so the HTTP layer stays a thin adapter over
the same functions the CLI scripts use -- ingestion at T2.2, asking after it.

Run it with:
    uvicorn app.api.main:app --reload
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import httpx
import psycopg
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app import config
from app.api.deps import get_connection, get_http_client
from app.api.schemas import AskRequest, AskResponse, IngestResponse
from app.pipeline import SUPPORTED_SUFFIXES, answer_question, ingest_text
from app.rag.chunking import DEFAULT_CHUNK_SIZE
from app.rag.embeddings import EmbeddingError
from app.rag.generation import GenerationError
from app.storage import supabase

# Uploads are read fully into memory and embedded inside the request, so the
# ceiling is deliberately modest until T6.3 moves this to a worker.
MAX_UPLOAD_BYTES = 2 * 1024 * 1024

TITLE = "rag-ai"
VERSION = "0.1.0"

DESCRIPTION = """\
Question answering over documents, using local Ollama embeddings and a
Supabase/pgvector store.

Interactive docs are at `/docs`; the OpenAPI schema is at `/openapi.json`.
"""

app = FastAPI(title=TITLE, version=VERSION, description=DESCRIPTION)

Connection = Annotated[psycopg.Connection, Depends(get_connection)]
HttpClient = Annotated[httpx.Client, Depends(get_http_client)]


@app.exception_handler(supabase.StorageError)
def _storage_unavailable(request: Request, exc: supabase.StorageError) -> JSONResponse:
    """The database is our dependency, not the caller's mistake."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(EmbeddingError)
@app.exception_handler(GenerationError)
def _ollama_unavailable(request: Request, exc: Exception) -> JSONResponse:
    """502: an upstream we depend on failed, rather than a bad request."""
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health", tags=["health"], summary="Liveness")
def health() -> dict[str, str]:
    """Whether the process is up. Deliberately cheap: no dependency calls.

    Use this for a restart check. /health/ready is the one that tells you
    whether the service can actually answer a question.
    """
    return {"status": "ok", "version": VERSION}


@app.get("/health/ready", tags=["health"], summary="Readiness")
def ready() -> dict[str, Any]:
    """Whether the dependencies this service needs are reachable.

    Reports rather than raises: a 200 with database.ok false is more useful
    to a human than a 503 with no detail. Callers should read the payload.
    """
    checks: dict[str, Any] = {}

    dsn = config.database_url()
    if dsn is None:
        checks["database"] = {"ok": False, "detail": "DATABASE_URL is not set"}
    else:
        try:
            with supabase.connect(dsn) as conn:
                checks["database"] = {"ok": True, "chunks": supabase.count_chunks(conn)}
        except supabase.StorageError as exc:
            checks["database"] = {"ok": False, "detail": str(exc)}

    try:
        response = httpx.get(f"{config.ollama_base_url()}/api/tags", timeout=5.0)
        response.raise_for_status()
        names = [model["name"] for model in response.json().get("models", [])]
        checks["ollama"] = {
            "ok": True,
            "embedding_model_present": any(
                name.startswith(config.embedding_model()) for name in names
            ),
            "chat_model_present": any(
                name.startswith(config.chat_model()) for name in names
            ),
        }
    except httpx.HTTPError as exc:
        checks["ollama"] = {"ok": False, "detail": str(exc)}

    ready = all(check["ok"] for check in checks.values())
    return {"ready": ready, "checks": checks}


@app.get("/config", tags=["health"], summary="Effective settings")
def settings() -> dict[str, str]:
    """The settings actually in force, for diagnosing environment problems.

    Never includes DATABASE_URL: it carries a password. Only whether it is
    configured is reported.
    """
    return config.describe()


def _decode_upload(raw: bytes, filename: str) -> str:
    """Validate an upload and return its text.

    Extensions are a claim, not evidence -- data/samples holds an IMG_0827.PNG
    that is actually a JPEG. For the text formats supported today, decoding is
    the real check: anything that is not UTF-8 text is rejected here rather
    than reaching the embedding model as mojibake.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type {suffix or filename!r}. "
                f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}. "
                "PDF and DOCX arrive at T2.5 and T2.6."
            ),
        )
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is {len(raw)} bytes; the limit is {MAX_UPLOAD_BYTES}.",
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=415,
            detail=(
                f"{filename} is not UTF-8 text despite its extension ({exc.reason})."
            ),
        ) from exc
    if not text.strip():
        raise HTTPException(status_code=422, detail=f"{filename} is empty.")
    return text


@app.post(
    "/documents",
    tags=["documents"],
    status_code=201,
    summary="Upload and ingest a document",
    response_model=IngestResponse,
)
def upload_document(
    conn: Connection,
    client: HttpClient,
    file: Annotated[UploadFile, File(description="A .txt or .md file")],
    # A form field, not a query parameter: with a multipart upload, -F is what
    # a caller reaches for, and a query parameter would be silently ignored.
    max_size: Annotated[int, Form(gt=0, le=8000)] = DEFAULT_CHUNK_SIZE,
) -> IngestResponse:
    """Chunk, embed and store an uploaded file.

    Re-uploading the same filename replaces that document's chunks rather
    than duplicating them, because ids are derived from the source name.
    """
    filename = file.filename or "upload"
    text = _decode_upload(file.file.read(), filename)
    result = ingest_text(
        text, source=filename, conn=conn, client=client, max_size=max_size
    )
    return IngestResponse.from_result(result)


@app.post(
    "/ask",
    tags=["ask"],
    summary="Answer a question from the stored documents",
    response_model=AskResponse,
)
def ask(request: AskRequest, conn: Connection, client: HttpClient) -> AskResponse:
    """Retrieve the closest chunks and answer from them.

    A refusal is reported as a 200 with is_refusal true: declining for lack
    of context is the correct outcome, not an error.
    """
    answer = answer_question(
        request.question, conn=conn, client=client, top_k=request.top_k
    )
    return AskResponse.from_answer(answer)
