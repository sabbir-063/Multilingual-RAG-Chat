import os
from loaders import load_doc, chunk_text
from gemini_client import GeminiClient
from embedding_client import HFEmbeddingClient
from vector_store import FaissStore, Chunk

class RAGPipeline:
    def __init__(self, api_key, store_dir="rag_store", chunk_size=1400, chunk_overlap=220):
        self.embedder = HFEmbeddingClient()
        self.gemini = GeminiClient(api_key)
        self.store = FaissStore(store_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def index_files(self, paths):
        all_chunks, texts = [], []
        cid = 0
        for p in paths:
            text = load_doc(p)
            spans = chunk_text(text, self.chunk_size, self.chunk_overlap)
            for start, end, ctext in spans:
                if len(ctext.strip()) < 50:
                    continue
                all_chunks.append(Chunk(cid, ctext, os.path.basename(p), start, end))
                texts.append(ctext)
                cid += 1

        # embeddings = self.gemini.embed_texts(texts)
        embeddings = self.embedder.embed_texts(texts)
        self.store.chunks = all_chunks
        self.store.build(embeddings)
        self.store.save()

    def load_index(self):
        self.store.load()

    def _make_prompt(self, query, retrieved, k=6):
        context = "\n".join([f"[{i+1}] {c.text}" for i, (c, _) in enumerate(retrieved[:k])])
        return f"""
                Answer the question using only the context below.
                If the answer isn't in context, say "I don’t know."

                Question: {query}
                Context: {context}
                """

    def ask(self, query, k=3):
        q_vec = self.embedder.embed_texts([query])[0]
        retrieved = self.store.search(q_vec, k)
        # print(retrieved)
        prompt = self._make_prompt(query, retrieved, k)
        answer = self.gemini.answer(prompt)
        return answer, retrieved
