# Gumloop RAG Assistant

A production-grade Retrieval-Augmented Generation (RAG) application that answers questions about [Gumloop](https://www.gumloop.com), a Vancouver-based AI automation startup.

## Stack

| Layer | Technology |
|---|---|
| LLM | Anthropic Claude |
| Vector Store | ChromaDB (persistent, local) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) |
| RAG Framework | LangChain LCEL |
| UI | Streamlit |

## Project Structure

```
RAG_PRODUCTION_QUALITY/
├── config/
│   ├── settings.py          # All config in one place (pydantic-settings)
│   └── secrets.env          # API keys — never commit this
├── data/
│   ├── chroma_db/           # Persisted vector index (auto-generated)
│   └── docs/                # Source documents for ingestion
├── scripts/
│   └── scrape_gumloop.py    # Collects public Gumloop content
├── src/
│   ├── components/
│   │   ├── embeddings.py    # Embedding model (cached singleton)
│   │   ├── retriever.py     # MMR retriever config
│   │   └── vector_store.py  # ChromaDB interface
│   ├── pipelines/
│   │   ├── ingestion_pipeline.py  # Load → chunk → embed → store
│   │   └── rag_pipeline.py        # Retrieve → prompt → LLM → answer
│   └── main.py              # Streamlit app (entry point)
└── tests/
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

Edit `config/secrets.env`:

```env
ANTHROPIC_API_KEY=
```

### 3. (Optional) Scrape fresh Gumloop data

```bash
python scripts/scrape_gumloop.py
```

A curated seed document is already included in `data/docs/`.

### 4. Build the vector index

```bash
python -m src.pipelines.ingestion_pipeline
```

To force a full re-index:
```bash
python -m src.pipelines.ingestion_pipeline --force
```

### 5. Launch the app

```bash
streamlit run src/main.py
```

## Architecture Notes

- **MMR Retrieval**: Uses Maximal Marginal Relevance instead of plain similarity search to ensure retrieved chunks are diverse and non-redundant.
- **Cached singletons**: Embedding model, vector store, and RAG chain are each loaded once and cached — no cold starts on every request.
- **Separated pipelines**: Ingestion and inference are fully decoupled. The app never rebuilds the index at runtime.
- **LCEL chain**: Uses LangChain Expression Language (the modern approach) rather than legacy `RetrievalQA`.