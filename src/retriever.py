import os
import re
import pickle
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

@dataclass
class Chunk:
    text: str
    page: int
    source: str

def clean_text(s: str) -> str:
    s = s.replace("\u00ad", "")
    return re.sub(r"\s+", " ", s).strip()

def split_sentences(text: str) -> List[str]:
    splitter = re.compile(r"(?<!\b[A-Z])(?<=[.?!])\s+(?=[A-Z0-9])")
    return [clean_text(p) for p in splitter.split(text) if p.strip()]

def extract_pdf_chunks(pdf_path: str, max_len: int = 512) -> List[Chunk]:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    chunks: List[Chunk] = []
    name = os.path.basename(pdf_path)

    for i, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if not text:
            continue

        buf: List[str] = []
        cur_len = 0
        for s in split_sentences(text):
            if cur_len + len(s) <= max_len:
                buf.append(s)
                cur_len += len(s)
            else:
                if buf:
                    chunks.append(Chunk(" ".join(buf), i, name))
                buf, cur_len = [s], len(s)
        if buf:
            chunks.append(Chunk(" ".join(buf), i, name))

    return chunks

@dataclass
class IndexArtifacts:
    index: faiss.Index
    meta: List[Chunk]
    dim: int

def load_embedder(model: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model, device="cpu")

def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

def build_index(chunks: List[Chunk], embedder: SentenceTransformer) -> IndexArtifacts:
    texts = [c.text for c in chunks]
    if not texts:
        raise ValueError("No text to index")

    emb = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    emb = normalize(emb.astype("float32"))

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    return IndexArtifacts(index, chunks, dim)

def save_index(art: IndexArtifacts, outdir: str, stem: str):
    os.makedirs(outdir, exist_ok=True)
    faiss.write_index(art.index, os.path.join(outdir, f"{stem}.faiss"))
    meta_list = [{"text": c.text, "page": c.page, "source": c.source} for c in art.meta]
    with open(os.path.join(outdir, f"{stem}.meta.pkl"), "wb") as f:
        pickle.dump({"meta": meta_list, "dim": art.dim}, f)

def load_index(outdir: str, stem: str) -> Optional[IndexArtifacts]:
    idx_path = os.path.join(outdir, f"{stem}.faiss")
    meta_path = os.path.join(outdir, f"{stem}.meta.pkl")
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        return None

    index = faiss.read_index(idx_path)
    with open(meta_path, "rb") as f:
        obj = pickle.load(f)
    meta = [Chunk(m["text"], m["page"], m["source"]) for m in obj["meta"]]
    return IndexArtifacts(index, meta, obj["dim"])

def search(query: str, top_k: int, embedder: SentenceTransformer, art: IndexArtifacts) -> List[Dict]:
    q = embedder.encode([query], convert_to_numpy=True)
    q = normalize(q.astype("float32"))

    D, I = art.index.search(q, top_k)
    results: List[Dict] = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        c = art.meta[idx]
        results.append({
            "score": float(score),
            "text": c.text,
            "page": c.page,
            "source": c.source,
        })
    return results
