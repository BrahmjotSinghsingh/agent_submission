from typing import List, Tuple

from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logger import get_logger
from app.rag.bm25 import bm25_retrieve
from app.rag.vectorstore import get_vectorstore

logger = get_logger(__name__)


def hybrid_retrieve(
    query: str,
    final_k: int | None = None,
    dense_k: int | None = None,
    bm25_k: int | None = None,
) -> List[Tuple[Document, float]]:
    """
    Hybrid retrieval combining dense (Chroma) and sparse (BM25) scores.

    Score formula:
        final_score = DENSE_WEIGHT * dense_score + BM25_WEIGHT * bm25_score

    Returns a list of (Document, score) tuples sorted by descending score.
    """
    settings = get_settings()
    final_k = final_k or settings.FINAL_K
    dense_k = dense_k or settings.DENSE_K
    bm25_k = bm25_k or settings.BM25_K

    vectorstore = get_vectorstore()

    # ── Dense retrieval ──────────────────────────────────────────────────────
    dense_hits = vectorstore.similarity_search_with_score(query, k=dense_k)
    dense_scores: dict[str, float] = {
        doc.metadata["entity_id"]: 1.0 / (1.0 + float(distance))
        for doc, distance in dense_hits
    }

    # ── BM25 retrieval ───────────────────────────────────────────────────────
    bm25_scores = bm25_retrieve(query, top_k=bm25_k)

    # ── Fusion ───────────────────────────────────────────────────────────────
    all_ids = set(dense_scores) | set(bm25_scores)

    # Build a lookup from entity_id → Document using the dense hits first
    docs_by_id: dict[str, Document] = {
        doc.metadata["entity_id"]: doc for doc, _ in dense_hits
    }

    # For BM25-only hits, grab from vectorstore by metadata filter
    missing_ids = all_ids - set(docs_by_id)
    if missing_ids:
        try:
            extra = vectorstore.get(
                where={"entity_id": {"$in": list(missing_ids)}},
                include=["documents", "metadatas"],
            )
            for content, meta in zip(extra["documents"], extra["metadatas"]):
                docs_by_id[meta["entity_id"]] = Document(
                    page_content=content, metadata=meta
                )
        except Exception as exc:
            logger.warning("Could not fetch BM25-only docs: %s", exc)

    fused: List[Tuple[Document, float]] = []
    for entity_id in all_ids:
        if entity_id not in docs_by_id:
            continue
        score = (
            settings.DENSE_WEIGHT * dense_scores.get(entity_id, 0.0)
            + settings.BM25_WEIGHT * bm25_scores.get(entity_id, 0.0)
        )
        fused.append((docs_by_id[entity_id], score))

    fused.sort(key=lambda x: x[1], reverse=True)
    top = fused[:final_k]

    logger.debug(
        "Hybrid retrieve: query=%r  dense=%d  bm25=%d  fused=%d  returned=%d",
        query[:60],
        len(dense_scores),
        len(bm25_scores),
        len(fused),
        len(top),
    )
    return top
