"""
Retriever configuration.

We use MMR (Maximal Marginal Relevance) instead of plain similarity search.

Why it matters:
  Plain similarity search can return k nearly-identical chunks — you get
  the same sentence paraphrased 6 times. MMR balances relevance with
  diversity, so each retrieved chunk actually adds new information.

  lambda_mult in settings controls the trade-off:
    1.0  →  pure similarity (like standard search)
    0.0  →  pure diversity (ignores relevance)
    0.6  →  our default: relevance-first, with diversity as a tiebreaker
"""

from langchain_core.vectorstores import VectorStoreRetriever
from loguru import logger

from config.settings import settings
from src.components.vector_store import get_vector_store


def get_retriever() -> VectorStoreRetriever:
    """
    Builds and returns an MMR-based retriever from the persisted vector store.
    """
    store = get_vector_store()
    logger.debug(
        f"Building MMR retriever | k={settings.retriever_k} "
        f"fetch_k={settings.retriever_fetch_k} "
        f"lambda={settings.mmr_lambda_mult}"
    )
    retriever = store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.retriever_k,
            "fetch_k": settings.retriever_fetch_k,
            "lambda_mult": settings.mmr_lambda_mult,
        },
    )
    return retriever