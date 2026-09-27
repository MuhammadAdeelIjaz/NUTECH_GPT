import time

import streamlit as st

from src import config
from src.llm import generate_answer
from src.retriever import Retriever

st.set_page_config(page_title=config.APP_TITLE, page_icon="🎓", layout="wide")

# ---------------------- Light custom styling ----------------------
st.markdown(
    """
    <style>
    .app-header {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .app-header h1 { margin: 0; font-size: 1.6rem; }
    .app-header p { margin: 0.2rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }
    .source-card {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
        background-color: #F8FAFC;
    }
    .source-card .fname { font-weight: 600; color: #1E3A8A; }
    .source-card .meta { font-size: 0.8rem; color: #64748B; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading index and embedding model...")
def load_retriever() -> Retriever:
    return Retriever()


retriever = load_retriever()

# ---------------------- Header ----------------------
st.markdown(
    f"""
    <div class="app-header">
        <h1>🎓 {config.APP_TITLE}</h1>
        <p>{config.APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------- Sidebar ----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    role = st.radio("I am a:", list(config.ROLE_DOC_TYPES.keys()))
    top_k = st.slider(
        "Sources to retrieve",
        min_value=3, max_value=10, value=config.DEFAULT_TOP_K,
        help="How many document chunks are pulled from the index and given "
             "to the assistant as context. Lower = faster, more focused. "
             "Higher = better for questions spanning multiple documents.",
    )

    st.divider()
    st.subheader("💡 Try asking")
    for q in config.SUGGESTED_QUESTIONS.get(role, []):
        if st.button(q, use_container_width=True, key=f"suggest_{q}"):
            st.session_state.pending_query = q

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    st.caption(
        "Answers are grounded only in indexed university documents and "
        "always show their sources for verification."
    )

# ---------------------- Session state ----------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def render_sources(chunks):
    """Renders the source cards. Called both right after generation and
    when redrawing chat history, so sources persist across reruns."""
    for i, c in enumerate(chunks, start=1):
        page_info = f" · page {c['page']}" if c.get("page") else ""
        confidence = max(0.0, min(1.0, c["score"]))
        st.markdown(
            f"""
            <div class="source-card">
                <span class="fname">[{i}] {c['source_file']}</span>
                <span class="meta">{page_info} · {c['doc_type']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(confidence, text=f"Relevance: {confidence:.0%}")


def process_query(query: str):
    """Runs retrieval + generation and appends BOTH turns to history,
    including the retrieved chunks — this is what makes sources persist
    after the next Streamlit rerun instead of vanishing."""
    st.session_state.history.append({"role": "user", "content": query})

    doc_types = config.ROLE_DOC_TYPES[role]
    with st.spinner("Searching indexed documents..."):
        chunks = retriever.search(query, top_k=top_k, doc_types=doc_types)

    if not chunks:
        answer = (
            "I couldn't find anything relevant to this question in the "
            "documents available to your role."
        )
        elapsed = None
    else:
        start = time.time()
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, chunks, role)
        elapsed = time.time() - start

    st.session_state.history.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks,      # <- persisted, not just shown once
        "elapsed": elapsed,
    })


# ---------------------- Handle new input BEFORE drawing history ----------------------
incoming_query = st.session_state.pending_query
st.session_state.pending_query = None

typed_query = st.chat_input("Ask a question...")
if typed_query:
    incoming_query = typed_query

if incoming_query:
    process_query(incoming_query)

# ---------------------- Draw full chat history (always, from state) ----------------------
if not st.session_state.history:
    st.info(
        f"👋 Welcome! Ask a question below, or try one of the suggestions "
        f"in the sidebar to see what {config.APP_TITLE} can help with."
    )

for msg in st.session_state.history:
    avatar = "🧑" if msg["role"] == "user" else "🎓"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        sources = msg.get("sources")
        if sources:
            if msg.get("elapsed") is not None:
                st.caption(f"⏱ Answered in {msg['elapsed']:.1f}s using {len(sources)} sources")
            with st.expander(f"📄 Sources ({len(sources)})", expanded=False):
                render_sources(sources)
