from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.logger import get_logger
from app.pipeline import build_index, load_models, run_pipeline


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        reranker, llm = load_models()
        vector_store, bm25, docs = build_index()
        app.state.docs = docs
        app.state.vector_store = vector_store
        app.state.bm25 = bm25
        app.state.reranker = reranker
        app.state.llm = llm
        logger.info("RAG System Ready")
        yield
    except Exception:
        logger.exception("Application startup failed")
        raise
    finally:
        logger.info("Shutdown")


app = FastAPI(lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


@app.get("/")
async def home():
    return {"status": "running"}


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, exc: RequestValidationError):
    messages = []

    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", []) if part != "body")
        message = error.get("msg", "Invalid request")
        if location:
            messages.append(f"{location}: {message}")
        else:
            messages.append(message)

    detail = "; ".join(messages) or "Invalid request body"
    logger.warning("Validation error on %s: %s", request.url.path, detail)
    return JSONResponse(status_code=422, content={"detail": detail, "error": "validation_error"})


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.warning("Value error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=400, content={"detail": str(exc), "error": "validation_error"})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    logger.error("Runtime error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": str(exc), "error": "system_error"})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Unexpected internal server error", "error": "internal_error"},
    )


@app.post("/ask")
async def ask(query: QueryRequest):
    query_text = query.query.strip()

    if not query_text:
        raise ValueError("Query cannot be empty.")

    answer, source = run_pipeline(
        query=query_text,
        vector_store=app.state.vector_store,
        bm25=app.state.bm25,
        reranker=app.state.reranker,
        llm=app.state.llm,
        docs=app.state.docs,
    )

    return {
        "answer": answer,
        "source": [
            {
                key: value
                for key, value in (getattr(doc, "metadata", {}) or {}).items()
                if key not in {"doc_id", "chunk_id", 'chunk_index'}
            }
            for doc in source
        ],
    }
