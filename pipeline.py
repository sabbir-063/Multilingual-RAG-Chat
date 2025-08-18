import os
from loaders import load_doc, chunk_text
from gemini_client import GeminiClient
from embedding_client import HFEmbeddingClient
from vector_store import FaissStore, Chunk
from text_processor import clean_text, detect_language
from guardrails import ContentGuardRails

class RAGPipeline:
    def __init__(self, api_key, store_dir="rag_store", chunk_size=1000, chunk_overlap=220):
        self.embedder = HFEmbeddingClient()
        self.gemini = GeminiClient(api_key)
        self.store = FaissStore(store_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.guardrails = ContentGuardRails()

    def index_files(self, paths):
        all_chunks, texts = [], []
        cid = 0
        for p in paths:
            # Load and clean the text based on language
            text = load_doc(p)
            cleaned_text = clean_text(text)
            
            # Chunk the cleaned text
            spans = chunk_text(cleaned_text, self.chunk_size, self.chunk_overlap)
            for start, end, ctext in spans:
                if len(ctext.strip()) < 50:
                    continue
                    
                # Store the language information with the chunk
                lang = detect_language(ctext)
                chunk = Chunk(cid, ctext, os.path.basename(p), start, end)
                chunk.language = lang  # Add language information to chunk metadata
                
                all_chunks.append(chunk)
                texts.append(ctext)
                cid += 1

        # embeddings = self.gemini.embed_texts(texts)
        embeddings = self.embedder.embed_texts(texts)
        self.store.chunks = all_chunks
        self.store.build(embeddings)
        self.store.save()

    def load_index(self):
        self.store.load()

    def _make_prompt(self, query, retrieved, k=3):
        context = "\n".join([f"[{i+1}] {c.text}" for i, (c, _) in enumerate(retrieved[:k])])
        return f"""
                Answer the question using only the context below.
                If the answer isn't in context, say "I don’t know."

                Question: {query}
                Context: {context}
                """

    def ask(self, query, k=3):
        # Apply safety check to the query
        is_safe, reason = self.guardrails.check_content_safety(query)
        if not is_safe:
            return "I apologize, but I cannot process that query as it may contain inappropriate content.", []
            
        # Get query embedding and search
        q_vec = self.embedder.embed_texts([query])[0]
        retrieved = self.store.search(q_vec, k)
        
        # Create prompt with safety guidelines
        prompt = self._make_prompt(query, retrieved, k)
        safe_prompt = self.guardrails.sanitize_prompt(prompt)
        
        # Get and sanitize the answer
        raw_answer = self.gemini.answer(safe_prompt)
        safe_answer, was_modified = self.guardrails.sanitize_response(raw_answer)
        
        return safe_answer, retrieved
