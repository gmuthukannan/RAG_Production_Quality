"""
Central configuration for the RAG application.
Uses pydantic-settings so every value can be overridden via environment variables --
no more hardcoded strings scattered across the codebase.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / "config" / "secrets.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Anthropic ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    temperature: float = 0.2          # Low temp = factual, grounded answers

    # ── Embeddings ─────────────────────────────────────────────────────────────
    # Free, local, no extra API key required.
    # Upgrade path: swap for "voyage-large-2" (Anthropic's recommended partner).
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── ChromaDB ───────────────────────────────────────────────────────────────
    chroma_persist_dir: str = str(ROOT_DIR / "data" / "chroma_db")
    chroma_collection_name: str = "gumloop_knowledge_base"

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retriever_k: int = 6              # How many chunks to retrieve
    retriever_fetch_k: int = 20       # MMR candidate pool size
    mmr_lambda_mult: float = 0.6      # 1.0 = pure similarity, 0.0 = pure diversity

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 100          # Overlap avoids losing context at boundaries

    # ── Data ───────────────────────────────────────────────────────────────────
    docs_dir: str = str(ROOT_DIR / "data" / "docs")

    # ── App ────────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"


# Singleton — import this everywhere instead of re-instantiating
settings = Settings()