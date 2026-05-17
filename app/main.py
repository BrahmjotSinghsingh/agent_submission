from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.logger import get_logger, setup_logging
from app.rag.bm25 import build_bm25_index
from app.rag.data_loader import build_documents, build_records, load_catalog
from app.rag.vectorstore import init_vectorstore

settings = get_settings()
setup_logging(debug=settings.DEBUG)
logger = get_logger(__name__)


# ─── Startup / Shutdown ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load data and build indexes on startup; clean up on shutdown."""
    logger.info("=== SHL RAG Agent starting up ===")

    # 1. Load and enrich catalog
    raw_data = load_catalog()
    records = build_records(raw_data)
    docs = build_documents(records)

    # 2. Build vector store (loads from disk if it already exists)
    init_vectorstore(docs)

    # 3. Build BM25 index (always in-memory)
    build_bm25_index(docs)

    logger.info("=== All indexes ready — serving requests ===")
    yield

    logger.info("=== SHL RAG Agent shutting down ===")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "A Retrieval-Augmented Generation (RAG) agent that recommends "
        "SHL assessments using hybrid retrieval (ChromaDB + BM25) "
        "and a Gemini LLM, orchestrated with LangGraph."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── Middleware ───────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────────────────────────

app.include_router(router)


@app.get("/", tags=["Utility"], summary="Root")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
    }
