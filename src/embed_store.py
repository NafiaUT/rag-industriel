"""
Étape 3 : Embeddings + Indexation hybride (FAISS Vectoriel + BM25 Mots-clés).

Charge les chunks générés par ingest.py, calcule leurs embeddings avec
un modèle local (sentence-transformers), construit l'index FAISS vectoriel
et l'index BM25 par mots-clés.

Usage:
    uv run python src/embed_store.py
"""

import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
INDEX_PATH = PROCESSED_DIR / "faiss.index"
BM25_PATH = PROCESSED_DIR / "bm25.pkl"
METADATA_PATH = PROCESSED_DIR / "metadata.pkl"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def tokenize(text: str) -> list[str]:
    """Tokenise un texte en mots minuscules pour l'indexation BM25."""
    return re.findall(r"\w+", text.lower())


def load_chunks() -> list[dict]:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks: list[dict]) -> None:
    texts = [c["text"] for c in chunks]

    # 1. Indexation Vectorielle Dense (FAISS)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f"Calcul des embeddings pour {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    # 2. Indexation par Mots-clés Sparse (BM25)
    print(f"Construction de l'index BM25 pour {len(texts)} chunks...")
    tokenized_corpus = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[OK] Index FAISS ({index.ntotal} vecteurs) et index BM25 générés dans {PROCESSED_DIR}")


def main():
    if not CHUNKS_PATH.exists():
        print("chunks.json introuvable. Lance d'abord: uv run python src/ingest.py")
        return
    chunks = load_chunks()
    if not chunks:
        print("Aucun chunk à indexer.")
        return
    build_index(chunks)


if __name__ == "__main__":
    main()

