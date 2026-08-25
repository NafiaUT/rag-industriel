"""
Interface Streamlit pour l'assistant documentaire intelligent.

Usage:
    uv run streamlit run app.py
"""

import json
from pathlib import Path

import streamlit as st

from src.embed_store import build_index
from src.ingest import RAW_DIR, process_all_pdfs
from src.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Assistant Documentaire Industriel",
    page_icon="🏭",
    layout="wide",
)

PROCESSED_CHUNKS = Path(__file__).resolve().parent / "data" / "processed" / "chunks.json"

# --- BARRE LATÉRALE (SIDEBAR) ---
st.sidebar.title("⚙️ Gestion du Corpus")

# 1. Upload de PDF
uploaded_files = st.sidebar.file_uploader(
    "📥 Téléverser des PDF",
    type=["pdf"],
    accept_multiple_files=True,
    help="Ajoutez vos fiches techniques et manuels industriels.",
)

if uploaded_files:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for uploaded_file in uploaded_files:
        save_path = RAW_DIR / uploaded_file.name
        if not save_path.exists():
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_count += 1
    if saved_count > 0:
        st.sidebar.success(f"{saved_count} fichier(s) sauvegardé(s) dans data/raw/")

# 2. Bouton de ré-indexation 1-clic
if st.sidebar.button("⚡ Ré-indexer les documents", use_container_width=True):
    with st.sidebar.status("Indexation en cours...", expanded=True) as status:
        st.write("Parsing PyMuPDF & découpage sémantique...")
        chunks = process_all_pdfs()
        st.write(f"{len(chunks)} chunks générés. Calcul des embeddings & BM25...")
        if chunks:
            build_index(chunks)
            st.cache_resource.clear()
            status.update(label="Indexation terminée avec succès !", state="complete", expanded=False)
            st.sidebar.success("Base documentaire mise à jour !")
            st.rerun()

# 3. Statistiques du corpus & Gestion des fichiers
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Statistiques du Corpus")

pdf_files = list(RAW_DIR.glob("*.pdf")) if RAW_DIR.exists() else []
st.sidebar.metric("Documents PDF au total", len(pdf_files))

chunk_count = 0
if PROCESSED_CHUNKS.exists():
    try:
        with open(PROCESSED_CHUNKS, "r", encoding="utf-8") as f:
            chunk_count = len(json.load(f))
    except Exception:
        chunk_count = 0
st.sidebar.metric("Chunks Indexés (FAISS + BM25)", chunk_count)

if pdf_files:
    with st.sidebar.expander(f"📁 Liste des PDF ({len(pdf_files)})"):
        for pdf_file in pdf_files:
            col1, col2 = st.columns([4, 1])
            col1.caption(f"📄 {pdf_file.name}")
            if col2.button("❌", key=f"del_{pdf_file.name}", help=f"Supprimer {pdf_file.name}"):
                pdf_file.unlink()
                st.sidebar.success(f"{pdf_file.name} supprimé.")
                st.rerun()

# 4. Effacer l'historique
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Effacer la discussion", use_container_width=True):
    st.session_state.history = []
    st.rerun()


# --- INTERFACE PRINCIPALE ---
st.title("🏭 Assistant Documentaire Intelligent — Environnement Industriel")
st.caption("RAG Hybride (FAISS + BM25 + Cross-Encoder) alimenté par Gemini 3.6 Flash")


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


try:
    pipeline = load_pipeline()
except FileNotFoundError:
    st.warning(
        "⚠️ Aucun index trouvé. Placez des PDF dans la barre latérale et cliquez sur '⚡ Ré-indexer les documents'."
    )
    st.stop()
except ValueError as e:
    st.error(f"🔑 Erreur de configuration : {e}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

question = st.chat_input("Posez votre question sur les documents techniques...")

if question:
    # Affichage immédiat du message utilisateur
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche hybride & génération de la réponse..."):
            result = pipeline.answer(question, history=st.session_state.history)

        st.markdown(result["answer"])

        # Affichage si la question a été reformulée pour le contexte multi-tours
        if result.get("search_query") and result["search_query"].strip() != question.strip():
            st.caption(f"🔍 *Recherche effectuée avec le contexte :* `{result['search_query']}`")

        with st.expander("📄 Sources utilisées (Recherche Hybride + Reranked)"):
            for s in result["sources"]:
                st.markdown(
                    f"**{s['source']}** — Page {s['page']} *(Score Reranker : {s['score']:.2f})*"
                )
                st.text(s["text"][:350] + ("..." if len(s["text"]) > 350 else ""))

    # Mise à jour de l'historique de discussion
    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": result["answer"]})

