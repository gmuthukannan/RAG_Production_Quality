"""
ChromaDB vector store interface.

Responsibilities:
  - Create / load a persistent Chroma collection
  - Add documents (called once during ingestion)
  - Expose the store for retrieval

Design note: we keep the Chroma client as a module-level singleton.
In production you'd likely swap this for a remote Chroma server or
a managed service (Pinecone, Weaviate), but the interface here stays
identical — only this file changes.
"""

from functools import lru_cache
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from loguru import logger

from config.settings import settings
from src.components.embeddings import get_embedding_model


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """
    Returns a persistent Chroma vector store, loading from disk if it exists.
    Safe to call at startup — will not re-embed documents.
    """
    logger.info(
        f"Connecting to ChromaDB at: {settings.chroma_persist_dir} "
        f"(collection: {settings.chroma_collection_name})"
    )
    store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=settings.chroma_persist_dir,
    )
    count = store._collection.count()
    logger.info(f"Vector store ready — {count} chunks indexed.")
    return store


def add_documents(documents: List[Document]) -> None:
    """
    Embeds and persists a list of LangChain Documents.
    Called exclusively from the ingestion pipeline.
    """
    store = get_vector_store()
    logger.info(f"Indexing {len(documents)} chunks into ChromaDB...")
    store.add_documents(documents)
    logger.info("Indexing complete.")


def collection_is_empty() -> bool:
    """Utility used by the ingestion pipeline to avoid double-indexing."""
    store = get_vector_store()
    return store._collection.count() == 0