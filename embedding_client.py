# embedding_client.py
from sentence_transformers import SentenceTransformer
import numpy as np

class HFEmbeddingClient:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        # Returns np.array (N, D)
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings
