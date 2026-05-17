import shutil
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logger import get_logger
from app.rag.embeddings import get_embeddings

logger = get_logger(__name__)

_vectorstore: Chroma | None = None


def build_vectorstore(docs: List[Document], reset: bool = False) -> Chroma:
    """Build (or load) the Chroma vector store from documents."""
    settings = get_settings()
    vector_dir = Path(settings.CHROMA_DIR)

    if reset and vector_dir.exists():
        logger.info("Resetting Chroma directory: %s", vector_dir)
        shutil.rmtree(vector_dir)

    if vector_dir.exists() and any(vector_dir.iterdir()):
        logger.info("Loading existing Chroma store from %s", vector_dir)
        store = Chroma(
            persist_directory=str(vector_dir),
            embedding_function=get_embeddings(),
            collection_name=settings.CHROMA_COLLECTION,
        )
    else:
        logger.info(
            "Indexing %d documents into Chroma at %s", len(docs), vector_dir
        )
        store = Chroma.from_documents(
            documents=docs,
            embedding=get_embeddings(),
            persist_directory=str(vector_dir),
            collection_name=settings.CHROMA_COLLECTION,
        )
        logger.info("Chroma index built successfully")

    return store


def get_vectorstore() -> Chroma:
    """Return the cached vector store (must be initialized first)."""
    global _vectorstore
    if _vectorstore is None:
        raise RuntimeError(
            "Vector store not initialised. Call init_vectorstore() on startup."
        )
    return _vectorstore


def init_vectorstore(docs: List[Document], reset: bool = False) -> None:
    """Initialize and cache the global vector store."""
    global _vectorstore
    _vectorstore = build_vectorstore(docs, reset=reset)
    logger.info("Global vector store ready")
