import os

APP_TITLE = "University Knowledge Assistant"

# ---- FAISS index location (committed to the repo, produced offline in Colab) ----
FAISS_INDEX_DIR = "faiss_index"
FAISS_INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(FAISS_INDEX_DIR, "metadata.json")

# ---- Must exactly match the embedding model used when the index was built ----
# (mismatched embedding model = mismatched vector space = broken retrieval)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ---- Groq LLM ----
GROQ_MODEL_NAME = "openai/gpt-oss-120b"

# ---- Retrieval defaults ----
DEFAULT_TOP_K = 5

# ---- Role -> which doc_type(s) from metadata that role is allowed to retrieve ----
ROLE_DOC_TYPES = {
    "Student": ["academic"],
    "Staff / Faculty": ["academic", "policy"],
}
