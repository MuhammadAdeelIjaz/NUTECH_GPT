import streamlit as st

from src import config
from src.llm import generate_answer
from src.retriever import Retriever

st.set_page_config(page_title=config.APP_TITLE, page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading index and embedding model...")
def load_retriever() -> Retriever:
    return Retriever()


retriever = load_retriever()

st.title(f"🎓 {config.APP_TITLE}")
st.caption(
    "Answers are grounded only in indexed university documents. "
    "Sources are shown with every answer for verification."
)

with st.sidebar:
    st.header("Settings")
    role = st.radio("I am a:", list(config.ROLE_DOC_TYPES.keys()))
    top_k = st.slider("Sources to retrieve", min_value=3, max_value=10, value=config.DEFAULT_TOP_K)
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question...")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    doc_types = config.ROLE_DOC_TYPES[role]

    with st.chat_message("assistant"):
        with st.spinner("Searching indexed documents..."):
            chunks = retriever.search(query, top_k=top_k, doc_types=doc_types)

        if not chunks:
            answer = (
                "I couldn't find anything relevant to this question in the "
                "indexed documents available to your role."
            )
            st.markdown(answer)
        else:
            with st.spinner("Generating answer..."):
                answer = generate_answer(query, chunks, role)
            st.markdown(answer)

            with st.expander(f"📄 Sources ({len(chunks)})"):
                for i, c in enumerate(chunks, start=1):
                    page_info = f", page {c['page']}" if c.get("page") else ""
                    st.markdown(
                        f"**[{i}]** `{c['source_file']}`{page_info} "
                        f"— *{c['doc_type']}* — similarity {c['score']:.3f}"
                    )

    st.session_state.history.append({"role": "assistant", "content": answer})
