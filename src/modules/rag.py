import numpy as np
import re
import faiss
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from src.core.config import settings
from src.core.logger import app_logger

class RAGEngine:
    def __init__(self):
        app_logger.info(f"Loading Embedding Model: {settings.EMBEDDING_MODEL_ID}")
        # This model WILL still use the GPU automatically if available
        self.embed_model = SentenceTransformer(settings.EMBEDDING_MODEL_ID, trust_remote_code=True)
        self.dimension = self.embed_model.get_sentence_embedding_dimension()

    def clean_ocr_text(self, text: str) -> str:
        text = re.sub(r'[^a-zA-Z0-9\s.,:/()-]', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def create_index(self, documents: List[Dict]) -> tuple[faiss.Index, List[Dict]]:
        """
        Takes raw OCR results, chunks them, and builds a FAISS index.
        Returns (index, metadata_map).
        """
        chunked_docs = []
        texts_to_embed = []

        # 1. Chunking Strategy
        for doc in documents:
            clean_text = self.clean_ocr_text(doc['text'])
            chunks = self._chunk_text(clean_text)
            for chunk in chunks:
                texts_to_embed.append(chunk)
                chunked_docs.append({
                    "filename": doc['filename'],
                    "content": chunk,
                    "full_path": doc['path']
                })

        if not texts_to_embed:
            return None, []

        # 2. Embedding 
        # Sentences are encoded on GPU (if torch sees it), then converted to numpy for FAISS
        embeddings = self.embed_model.encode(texts_to_embed, convert_to_numpy=True, normalize_embeddings=True)

        # 3. FAISS Indexing (CPU Mode)
        # IndexFlatIP is exact search. It is extremely fast on CPU for < 100k vectors.
        index = faiss.IndexFlatIP(self.dimension)
        index.add(embeddings)
        
        return index, chunked_docs

    def search(self, index: faiss.Index, metadata: List[Dict], query: str, k: int = 5):
        """
        Embeds the query and retrieves top-k chunks.
        """
        if index is None or index.ntotal == 0:
            return []

        # Query embedding happens on GPU (via self.embed_model)
        query_vec = self.embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Search happens on CPU (Fast & Stable)
        distances, indices = index.search(query_vec, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append({
                    "score": float(distances[0][i]),
                    "chunk": metadata[idx]
                })
        return results

    def _chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

rag_engine = RAGEngine()