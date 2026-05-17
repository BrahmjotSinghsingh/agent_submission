from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embedding model."""
    settings = get_settings()
    logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": settings.EMBEDDING_DEVICE},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.EMBEDDING_BATCH_SIZE,
        },
    )

    logger.info("Embedding model loaded")
    return embeddings
