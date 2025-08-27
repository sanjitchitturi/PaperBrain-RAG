"""
rag_pipeline.py
---------------
A thin wrapper to manage PDFs, indexes, and queries.
"""

import os
from typing import List, Dict

from sentence_transformers import SentenceTransformer

from .retriever import (
    extract_pdf_chunks,
    load_embedder,
    build_index,
    load_index,
    save_index,
    search,
    IndexArtifacts,
    Chunk,
)

class RagPipeline:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.manuals_dir = os.path.join(data_dir, "manuals")
        self.index_dir = os.path.join(data_dir, "indexes")
        os.makedirs(self.manuals_dir, exist_ok=True)
        os.makedirs(self.index_dir, exist_ok=True)

        self.embedder: SentenceTransformer = load_embedder()
        self.cache: dict[str, IndexArtifacts] = {}

    def _stem(self, pdf_path: str) -> str:
        return os.path.splitext(os.path.basename(pdf_path))[0]

    def ensure_index(self, pdf_path: str, rebuild: bool = False) -> IndexArtifacts:
        stem = self._stem(pdf_path)
        if not rebuild and stem in self.cache:
            return self.cache[stem]

        if not rebuild:
            cached = load_index(self.index_dir, stem)
            if cached:
                self.cache[stem] = cached
                return cached

        # Build fresh
        chunks = extract_pdf_chunks(pdf_path)
        art = build_index(chunks, self.embedder)
        save_index(art, self.index_dir, stem)
        self.cache[stem] = art
        return art

    def ask(self, pdf_path: str, question: str, k: int = 5) -> List[Dict]:
        art = self.ensure_index(pdf_path)
        return search(question, k, self.embedder, art)
