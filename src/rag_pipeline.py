"""
Étape 4 & 5 : Retrieval Hybride (FAISS + BM25) + Reranking (Cross-Encoder) + Génération Gemini.

1. Recherche Dense avec FAISS (Top-20).
2. Recherche Sparse par mots-clés avec BM25 (Top-20).
3. Fusion RRF (Reciprocal Rank Fusion) pour combiner les rangs.
4. Reranking avec Cross-Encoder ('cross-encoder/ms-marco-MiniLM-L-6-v2').
5. Génération de réponse sourcée avec Gemini.
"""

import os
import pickle
import re
from pathlib import Path

import faiss
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# Prise en charge à la fois du nouveau SDK (google-genai) et du SDK classique (google-generativeai)
try:
    from google import genai
    from google.genai import types
    USE_NEW_SDK = True
except (ImportError, AttributeError):
    import google.generativeai as genai
    USE_NEW_SDK = False

load_dotenv()

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
INDEX_PATH = PROCESSED_DIR / "faiss.index"
BM25_PATH = PROCESSED_DIR / "bm25.pkl"
METADATA_PATH = PROCESSED_DIR / "metadata.pkl"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

SYSTEM_PROMPT = """Tu es un assistant documentaire pour un environnement industriel.
Réponds de manière complète, détaillée et structurée à partir des extraits de documents fournis dans le contexte. Ne raccourcis pas tes explications.
Si l'information n'est pas présente dans le contexte, dis clairement que tu ne sais pas plutôt que d'inventer une réponse. Cite systématiquement le document et la page source de l'information utilisée."""


def tokenize(text: str) -> list[str]:
    """Tokenise un texte en mots minuscules pour la recherche BM25."""
    return re.findall(r"\w+", text.lower())


class RAGPipeline:
    def __init__(self, top_k: int = 7):
        self.top_k = top_k
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_PATH))

        # Chargement de l'index BM25 et des métadonnées
        with open(BM25_PATH, "rb") as f:
            self.bm25: BM25Okapi = pickle.load(f)

        with open(METADATA_PATH, "rb") as f:
            self.chunks: list[dict] = pickle.load(f)

        # Initialisation du Cross-Encoder pour le Reranking
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "La clé GEMINI_API_KEY est introuvable. Veuillez définir GEMINI_API_KEY dans votre fichier .env ou vos variables d'environnement."
            )

        if USE_NEW_SDK:
            self.client = genai.Client(api_key=api_key)
        else:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(
                model_name="gemini-3.6-flash",
                system_instruction=SYSTEM_PROMPT,
            )

    def retrieve_dense(self, query: str, top_n: int = 20) -> list[tuple[int, float]]:
        """Recherche vectorielle dense avec FAISS."""
        query_vec = self.embed_model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, top_n)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((int(idx), float(score)))
        return results

    def retrieve_sparse(self, query: str, top_n: int = 20) -> list[tuple[int, float]]:
        """Recherche par mots-clés sparse avec BM25."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = scores.argsort()[-top_n:][::-1]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((int(idx), score))
        return results

    def rrf_fusion(
        self, dense_results: list[tuple[int, float]], sparse_results: list[tuple[int, float]], k: int = 60
    ) -> list[dict]:
        """Fusionne les classements Dense (FAISS) et Sparse (BM25) via RRF (Reciprocal Rank Fusion)."""
        rrf_scores = {}

        # Calcul RRF pour FAISS
        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank + 1))

        # Calcul RRF pour BM25
        for rank, (idx, _) in enumerate(sparse_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (k + rank + 1))

        # Tri des chunks par score RRF décroissant
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        merged_chunks = []
        for idx, rrf_score in sorted_indices:
            chunk_copy = {**self.chunks[idx], "rrf_score": rrf_score}
            merged_chunks.append(chunk_copy)

        return merged_chunks

    def rerank(self, query: str, candidates: list[dict], top_k: int = 7) -> list[dict]:
        """Réordonne les candidats avec le modèle Cross-Encoder et retourne les top_k meilleurs."""
        if not candidates:
            return []

        # Construction des paires (question, passage) pour le Cross-Encoder
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)

        for chunk, score in zip(candidates, scores):
            chunk["score"] = float(score)

        # Tri par score Cross-Encoder décroissant
        reranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]

    def retrieve(self, query: str) -> list[dict]:
        """Exécute la recherche hybride (FAISS + BM25) suivie du Reranking par Cross-Encoder."""
        dense_hits = self.retrieve_dense(query, top_n=20)
        sparse_hits = self.retrieve_sparse(query, top_n=20)

        # Fusion RRF des Top-20 dense et sparse
        fused_candidates = self.rrf_fusion(dense_hits, sparse_hits)[:15]

        # Reranking Cross-Encoder sur les 15 meilleurs candidats
        final_sources = self.rerank(query, fused_candidates, top_k=self.top_k)
        return final_sources

    def build_context(self, retrieved_chunks: list[dict]) -> str:
        parts = []
        for c in retrieved_chunks:
            parts.append(f"[Source: {c['source']}, page {c['page']}]\n{c['text']}")
        return "\n\n---\n\n".join(parts)

    def condense_question(self, history: list[dict], query: str) -> str:
        """Formule une question autonome en intégrant le contexte des échanges récents."""
        if not history:
            return query

        recent_history = history[-4:]
        history_formatted = "\n".join(
            f"{'Utilisateur' if h['role'] == 'user' else 'Assistant'}: {h['content']}"
            for h in recent_history
        )

        prompt = f"""Étant donné la discussion suivante et la nouvelle question de l'utilisateur, reformule la nouvelle question pour qu'elle soit complètement autonome (en incluant le nom de la machine, de la pièce ou de la procédure mentionnée précédemment).
Ne réponds PAS à la question, retourne UNIQUEMENT la question reformulée.

Historique :
{history_formatted}

Nouvelle question : {query}
Question autonome reformulée :"""

        try:
            if USE_NEW_SDK:
                res = self.client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(max_output_tokens=300),
                )
                standalone = res.text.strip() if res.text else query
            else:
                res = self.client.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": 300},
                )
                standalone = res.text.strip() if res.text else query
            return standalone if standalone else query
        except Exception:
            return query

    def answer(self, query: str, history: list[dict] = None) -> dict:
        search_query = self.condense_question(history, query) if history else query
        retrieved = self.retrieve(search_query)
        context = self.build_context(retrieved)

        user_message = f"""Contexte documentaire :

{context}

Question : {query}"""

        if USE_NEW_SDK:
            response = self.client.models.generate_content(
                model="gemini-3.6-flash",
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=8192,
                ),
            )
            answer_text = response.text if response.text else ""
        else:
            response = self.client.generate_content(
                user_message,
                generation_config={"max_output_tokens": 8192},
            )
            answer_text = response.text if response.text else ""

        return {
            "answer": answer_text,
            "sources": retrieved,
            "search_query": search_query,
        }


if __name__ == "__main__":
    pipeline = RAGPipeline()
    question = input("Question : ")
    result = pipeline.answer(question)
    print("\n--- Réponse ---")
    print(result["answer"])
    print("\n--- Sources Hybrides (Reranked) ---")
    for s in result["sources"]:
        print(f"- {s['source']} (page {s['page']}, score Reranker={s['score']:.2f})")



