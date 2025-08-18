import os
import faiss
import pickle
import numpy as np
from dataclasses import dataclass

@dataclass
class Chunk:
    id: int
    text: str
    source: str
    start_char: int
    end_char: int
    language: str = 'english'  # Can be 'english', 'bangla', or 'mixed'

class FaissStore:
    def __init__(self, store_dir="rag_store"):
        self.store_dir = store_dir
        self.index = None
        self.chunks = []

    @property
    def index_path(self):
        return os.path.join(self.store_dir, "index.faiss")

    @property
    def meta_path(self):
        return os.path.join(self.store_dir, "chunks.pkl")

    def build(self, embeddings: np.ndarray):
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def save(self):
        os.makedirs(self.store_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.chunks = pickle.load(f)

    def search(self, query_vec: np.ndarray, k=3):
        q = query_vec.astype("float32")[None, :]
        faiss.normalize_L2(q)
        scores, idxs = self.index.search(q, k)
        out = []
        for i, s in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            out.append((self.chunks[i], float(s)))
        return out
