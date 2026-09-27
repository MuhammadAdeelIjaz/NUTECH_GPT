import os

APP_TITLE = "NUTECH_GPT"
APP_SUBTITLE = "Your academic knowledge assistant & policy advisor"

SUGGESTED_QUESTIONS = {
    "Student": [
        "What is the hostel policy and what are the current hostel charges?",
        "What support is available for students with disabilities?",
        "What are the rules on organizing a club or society?",
        "What is the discipline policy for engineering students?",
        "Can I pay my semester/hostel fee in installments?",
    ],
    "Staff / Faculty": [
        "What is the faculty leave policy?",
        "What is the internal employment policy?",
        "How does the Research Publication Award Policy work?",
        "What does the HR policy say about [topic]?",
        "What is covered under the NUTECH Act?",
    ],
}

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

# ---- Role -> which audience tag (from metadata) that role is allowed to retrieve ----
# A chunk is retrievable if this tag appears in the chunk's "audience" list
# (most chunks list one or both — see AUDIENCE_MAP in the indexing notebook).
ROLE_AUDIENCE = {
    "Student": "student",
    "Staff / Faculty": "staff",
}
