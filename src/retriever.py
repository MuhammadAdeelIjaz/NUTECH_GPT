import json

import faiss
from sentence_transformers import SentenceTransformer

from . import config


class Retriever:
    """Loads the prebuilt FAISS index + metadata (from the offline Colab
    indexing notebook) and performs cosine-similarity search, with optional
    filtering by audience (student / staff) based on the caller's role.
    """

    def __init__(self):
        self.index = faiss.read_index(config.FAISS_INDEX_PATH)
        with open(config.METADATA_PATH, "r") as f:
            self.metadata = json.load(f)
        # Must be the same model used to build the index, or similarity
        # scores are meaningless (vectors would live in different spaces).
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    def search(self, query: str, top_k: int = config.DEFAULT_TOP_K, audience: str = None):
        query_vec = self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        # Over-fetch when filtering by audience so we still end up with
        # top_k relevant results after discarding chunks not meant for this role.
        fetch_k = top_k * 4 if audience else top_k
        fetch_k = min(fetch_k, self.index.ntotal) or 1

        scores, indices = self.index.search(query_vec, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            if audience and audience not in meta.get("audience", []):
                continue
            results.append({**meta, "score": float(score)})
            if len(results) >= top_k:
                break
        return results

    def list_documents(self, audience: str = None):
        """Returns sorted (source_file, audience_list) pairs for every
        indexed document, optionally filtered to ones visible to `audience`.
        Used to show users what's actually available to ask about."""
        docs = {}
        for m in self.metadata:
            docs.setdefault(m["source_file"], set()).update(m.get("audience", []))
        items = sorted(docs.items())
        if audience:
            items = [(fname, aud) for fname, aud in items if audience in aud]
        return items
