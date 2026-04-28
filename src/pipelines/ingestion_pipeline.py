"""
Ingestion Pipeline

Orchestrates the one-time (or on-demand) process of:
  1. Loading raw documents from the docs directory
  2. Splitting them into chunks
  3. Embedding and persisting them to ChromaDB

Run this script directly to (re)build the index:
    python -m src.pipelines.ingestion_pipeline

Design principle: ingestion is always separate from inference.
Your app should never be rebuilding the index on a user's request.
"""

import sys
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

# ── Allow running as __main__ from the project root ───────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from config.settings import settings
from src.components.vector_store import add_documents, collection_is_empty, get_vector_store


def load_documents():
    """
    Loads all .txt and .md files from the docs directory.
    DirectoryLoader with glob pattern lets us add new file types trivially.
    """
    docs_path = Path(settings.docs_dir)
    if not docs_path.exists() or not any(docs_path.iterdir()):
        raise FileNotFoundError(
            f"No documents found in {docs_path}. "
            "Run scripts/scrape_gumloop.py first."
        )

    loader = DirectoryLoader(
        str(docs_path),
        glob="**/*.{txt,md}",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} raw document(s) from {docs_path}")
    return docs


def split_documents(documents):
    """
    Splits documents into overlapping chunks.

    RecursiveCharacterTextSplitter is the standard choice — it tries to split
    on paragraph breaks, then sentences, then words, only falling back to
    hard character splits as a last resort. This preserves semantic coherence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        f"Split {len(documents)} doc(s) into {len(chunks)} chunks "
        f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )
    return chunks


def run_ingestion(force: bool = False) -> None:
    """
    Full ingestion run.
    Skips if the collection already has data (use force=True to re-index).
    """
    if not force and not collection_is_empty():
        count = get_vector_store()._collection.count()
        logger.info(
            f"Collection already has {count} chunks. "
            "Skipping ingestion. Pass force=True to re-index."
        )
        return

    logger.info("Starting ingestion pipeline...")
    documents = load_documents()
    chunks = split_documents(documents)
    add_documents(chunks)
    logger.success(f"Ingestion complete. {len(chunks)} chunks indexed.")


if __name__ == "__main__":
    force_reindex = "--force" in sys.argv
    run_ingestion(force=force_reindex)