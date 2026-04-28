"""
Gumloop RAG Assistant — Streamlit UI

Entry point for the application.
Run with:  streamlit run src/main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from loguru import logger

from config.settings import settings
from src.pipelines.ingestion_pipeline import run_ingestion
from src.pipelines.rag_pipeline import answer


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Gumloop Assistant",
    page_icon="🔁",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ── Startup — ensure index exists ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Initialising knowledge base...")
def initialise():
    """
    Runs ingestion once per Streamlit session (cached across reruns).
    st.cache_resource means this function body executes only once
    even if the user sends 100 messages.
    """
    run_ingestion()
    logger.info("App initialised.")


initialise()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://framerusercontent.com/images/GWHkyGSBO6bAMDMxWTomNFRHU.png",
        width=180,
    )
    st.markdown("## About")
    st.markdown(
        "This assistant answers questions about **Gumloop**, "
        "a Vancouver-based AI workflow automation platform — "
        "using Retrieval-Augmented Generation (RAG).\n\n"
        "Answers are grounded exclusively in curated documentation; "
        "the model won't hallucinate facts it wasn't given."
    )
    st.divider()
    st.markdown("**Stack**")
    st.markdown(
        "- 🧠 LLM: Claude (Anthropic)\n"
        "- 📦 Vector DB: ChromaDB\n"
        "- 🔗 Framework: LangChain (LCEL)\n"
        "- 🖥 UI: Streamlit"
    )
    st.divider()
    st.caption(f"Model: `{settings.claude_model}`")
    st.caption(f"Env: `{settings.app_env}`")


# ── Chat UI ───────────────────────────────────────────────────────────────────

st.title("🔁 Gumloop Assistant")
st.caption("Ask anything about Gumloop — what it does, how it works, who it's for.")

# Initialise chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I'm a RAG-powered assistant trained on Gumloop's documentation. "
                "Ask me anything about the platform — features, use cases, pricing, the team, and more."
            ),
        }
    ]

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user input
if user_input := st.chat_input("Ask about Gumloop..."):
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and stream the assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reasoning..."):
            try:
                response = answer(user_input)
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                error_msg = f"⚠️ Something went wrong: {e}"
                st.error(error_msg)
                logger.exception("Error during RAG chain invocation")