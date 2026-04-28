"""
RAG Inference Pipeline

Builds the end-to-end retrieval-augmented generation chain using LangChain
Expression Language (LCEL) — the modern, composable approach replacing
legacy chains like RetrievalQA.

Chain flow:
    User question
        → Retriever (MMR) → relevant chunks
        → Prompt template (question + context)
        → Claude
        → Parsed string answer
"""

from functools import lru_cache
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from loguru import logger

from config.settings import settings
from src.components.retriever import get_retriever


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions about \
Gumloop, a Vancouver-based AI workflow automation startup.

Answer using ONLY the information provided in the context below. \
If the context doesn't contain enough information to answer confidently, \
say so clearly — do not fabricate details.

Be concise, accurate, and professional. When relevant, mention specific \
features, use cases, or facts from the context to support your answer.

Context:
{context}"""

HUMAN_PROMPT = "{question}"

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_context(documents: List[Document]) -> str:
    """
    Joins retrieved chunks into a single context string.
    Includes the source filename so Claude can implicitly attribute claims.
    """
    parts = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Chunk {i} | source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Chain factory ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_rag_chain():
    """
    Assembles and returns the LCEL RAG chain.
    Cached — the chain is stateless and safe to reuse across requests.
    """
    logger.info(f"Initialising RAG chain with model: {settings.claude_model}")

    llm = ChatAnthropic(
        model=settings.claude_model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )

    retriever = get_retriever()

    # RunnableParallel runs retrieval and question passthrough simultaneously
    chain = (
        RunnableParallel(
            {
                "context": retriever | format_context,
                "question": RunnablePassthrough(),
            }
        )
        | PROMPT
        | llm
        | StrOutputParser()
    )

    logger.info("RAG chain ready.")
    return chain


def answer(question: str) -> str:
    """
    Public interface — takes a plain question string, returns an answer string.
    This is the only function the Streamlit app needs to call.
    """
    logger.info(f"Query received: {question!r}")
    chain = get_rag_chain()
    response = chain.invoke(question)
    logger.debug(f"Response length: {len(response)} chars")
    return response