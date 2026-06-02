"""Real retriever over an embedded guideline corpus (FAISS + sentence-transformers).

Build the index first with `corpus/build_index.py`, then:
    from rag_retriever import FaissRetriever
    r = FaissRetriever.load("corpus/index")
    r.retrieve("how much fiber per day?", k=4)

Requires: `pip install sentence-transformers faiss-cpu`.
Matches the Retriever protocol in retrieval.py (retrieve(query, k, item_id)).
"""
from __future__ import annotations

import json
import os
from typing import List

from retrieval import Passage

_DEFAULT_EMBED = "sentence-transformers/all-MiniLM-L6-v2"


class FaissRetriever:
    def __init__(self, index, passages: List[Passage], embed_model):
        self._index = index
        self._passages = passages
        self._embed = embed_model

    @classmethod
    def load(cls, index_dir: str, embed_model_name: str = _DEFAULT_EMBED):
        import faiss
        from sentence_transformers import SentenceTransformer
        index = faiss.read_index(os.path.join(index_dir, "corpus.faiss"))
        with open(os.path.join(index_dir, "passages.json"), encoding="utf-8") as f:
            raw = json.load(f)
        passages = [Passage(**p) for p in raw]
        embed = SentenceTransformer(embed_model_name)
        return cls(index, passages, embed)

    def retrieve(self, query: str, k: int = 4,
                 item_id: str | None = None) -> List[Passage]:
        import numpy as np
        q = self._embed.encode([query], normalize_embeddings=True)
        scores, idx = self._index.search(np.asarray(q, dtype="float32"), k)
        out = []
        for j in idx[0]:
            if 0 <= j < len(self._passages):
                out.append(self._passages[j])
        return out
