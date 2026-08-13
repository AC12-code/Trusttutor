"""Chunk timestamped segments and retrieve the most relevant ones.

Two backends:
  * tfidf  (default): scikit-learn, no model download, instant, deterministic.
  * faiss           : sentence-transformers + faiss-cpu (matches the resume
                      stack). Flip with RETRIEVER=faiss.

Both return (chunk_text, start_timestamp, score) so the answer can cite time.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
from .config import settings
from .ingest import Segment


@dataclass
class Chunk:
    text: str
    start: float

    def ts(self) -> str:
        m, s = divmod(int(self.start), 60)
        return f"{m:02d}:{s:02d}"


def chunk_segments(segs: List[Segment], words_per_chunk: int | None = None) -> List[Chunk]:
    n = words_per_chunk or settings.chunk_words
    chunks: List[Chunk] = []
    buf, buf_start, count = [], None, 0
    for s in segs:
        if buf_start is None:
            buf_start = s.start
        buf.append(s.text)
        count += len(s.text.split())
        if count >= n:
            chunks.append(Chunk(" ".join(buf), buf_start))
            buf, buf_start, count = [], None, 0
    if buf:
        chunks.append(Chunk(" ".join(buf), buf_start or 0.0))
    return chunks


class Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.backend = settings.retriever.lower()
        if self.backend == "faiss":
            self._init_faiss()
        else:
            self._init_tfidf()

    # -- TF-IDF backend --
    def _init_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(stop_words="english")
        self._matrix = self._vec.fit_transform([c.text for c in self.chunks])

    def _search_tfidf(self, query: str, k: int):
        from sklearn.metrics.pairwise import cosine_similarity
        qv = self._vec.transform([query])
        sims = cosine_similarity(qv, self._matrix)[0]
        idx = sims.argsort()[::-1][:k]
        return [(self.chunks[i], float(sims[i])) for i in idx]

    # -- FAISS backend --
    def _init_faiss(self):
        from sentence_transformers import SentenceTransformer
        import faiss, numpy as np
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        embs = self._model.encode([c.text for c in self.chunks], normalize_embeddings=True)
        self._np = np
        self._index = faiss.IndexFlatIP(embs.shape[1])
        self._index.add(embs.astype("float32"))

    def _search_faiss(self, query: str, k: int):
        qv = self._model.encode([query], normalize_embeddings=True).astype("float32")
        scores, idx = self._index.search(qv, k)
        return [(self.chunks[i], float(scores[0][j])) for j, i in enumerate(idx[0]) if i != -1]

    def search(self, query: str, k: int | None = None) -> List[Tuple[Chunk, float]]:
        k = k or settings.top_k
        if self.backend == "faiss":
            return self._search_faiss(query, k)
        return self._search_tfidf(query, k)
