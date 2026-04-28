"""
Embedding model wrapper.

We use HuggingFace sentence-transformers locally:
  - Zero extra cost / no API key needed for embeddings
  - Fully reproducible (same model version = same vectors)
  - Upgrade path: swap for VoyageAIEmbeddings for production scale

The factory function makes it trivial to swap the backend later
without touching any pipeline code.
"""

from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger

from config.settings import settings


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns a cached embedding model instance.
    lru_cache ensures the model is only loaded from disk once per process —
    loading sentence-transformers on every request would be a silent perf killer.
    """
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # Cosine similarity friendly
    )
    logger.info("Embedding model loaded successfully.")
    return embeddings