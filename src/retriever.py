import json

import faiss
from sentence_transformers import SentenceTransformer

from . import config


class Retriever:
    """Loads the prebuilt FAISS index + metadata (from the offline Colab
    indexing notebook) and performs cosine-similarity search, with optional
    filtering by doc_type (academic / policy) based on the caller's role.
    """

    def __init__(self):
        self.index = faiss.read_index(config.FAISS_INDEX_PATH)
        with open(config.METADATA_PATH, "r") as f:
            self.metadata = json.load(f)
        # Must be the same model used to build the index, or similarity
        # scores are meaningless (vectors would live in different spaces).
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K, doc_types=None):
        query_vec = self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        # Over-fetch when filtering by doc_type so we still end up with
        # top_k relevant results after discarding the wrong type.
        fetch_k = top_k * 4 if doc_types else top_k
        fetch_k = min(fetch_k, self.index.ntotal) or 1

        scores, indices = self.index.search(query_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            if doc_types and meta.get("doc_type") not in doc_types:
                continue
            results.append({**meta, "score": float(score)})
            if len(results) >= top_k:
                break
        return results
