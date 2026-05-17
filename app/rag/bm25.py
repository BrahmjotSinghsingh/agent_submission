import re
from typing import Dict, List

import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.core.logger import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9+#.\-]+")

_bm25: BM25Okapi | None = None
_bm25_docs: List[Document] = []


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(str(text).lower())


def build_bm25_index(docs: List[Document]) -> None:
    """Build and cache the BM25 index over document page_content."""
    global _bm25, _bm25_docs
    _bm25_docs = docs
    corpus = [tokenize(doc.page_content) for doc in docs]
    _bm25 = BM25Okapi(corpus)
    logger.info("BM25 index built over %d documents", len(docs))


def bm25_retrieve(
    query: str, top_k: int = 40
) -> Dict[str, float]:
    """Return {entity_id: normalised_score} for the top-k BM25 hits."""
    if _bm25 is None:
        raise RuntimeError("BM25 index not initialised. Call build_bm25_index() first.")

    tokens = tokenize(query)
    raw_scores = _bm25.get_scores(tokens)
    top_idx = np.argsort(raw_scores)[::-1][:top_k]
    top_vals = [float(raw_scores[i]) for i in top_idx]

    # Min-max normalisation
    lo, hi = min(top_vals), max(top_vals)
    if lo == hi:
        normed = [1.0] * len(top_vals)
    else:
        normed = [(v - lo) / (hi - lo) for v in top_vals]

    return {
        _bm25_docs[int(i)].metadata["entity_id"]: score
        for i, score in zip(top_idx, normed)
    }
