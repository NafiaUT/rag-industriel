"""
Étape 1 & 2 : Ingestion intelligente des PDF avec PyMuPDF + Découpage sémantique.

Lit tous les PDF dans data/raw/, extrait le texte et les tableaux au format Markdown,
effectue un nettoyage (en-têtes, pieds de page répétitifs), puis applique un découpage
récursif intelligent (paragraphes, phrases) avant de sauvegarder le résultat dans
data/processed/chunks.json.

Usage:
    uv run python src/ingest.py
"""

import json
import re
from pathlib import Path

import pymupdf
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

CHUNK_SIZE = 1000  # Taille cible des chunks en caractères
CHUNK_OVERLAP = 200  # Recouvrement entre chunks consécutifs
SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


def clean_text(text: str) -> str:
    """Nettoie le texte extrait d'une page PDF."""
    if not text:
        return ""
    # Supprime les numéros de page isolés en bas ou haut (ex: "Page 1 sur 10", "Page 3")
    text = re.sub(r"(?i)^\s*page\s+\d+(\s+sur\s+\d+)?\s*$", "", text, flags=re.MULTILINE)
    # Normalise les espaces multiples (garde les sauts de ligne)
    text = re.sub(r"[ \t]+", " ", text)
    # Réduit les suites de plus de 2 sauts de ligne à 2 sauts de ligne max
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_page_content(page: pymupdf.Page) -> str:
    """Extraie le texte d'une page en intégrant les tableaux au format Markdown."""
    blocks = []

    # 1. Extraction et conversion des tableaux en Markdown si disponibles
    try:
        tables = page.find_tables()
        for table in tables:
            md_table = table.to_markdown()
            if md_table and md_table.strip():
                blocks.append(f"\n\n{md_table.strip()}\n\n")
    except Exception:
        pass

    # 2. Extraction du texte de la page
    raw_text = page.get_text("text")
    cleaned = clean_text(raw_text)
    if cleaned:
        blocks.append(cleaned)

    return "\n\n".join(blocks)


def extract_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extrait le contenu nettoyé page par page pour un document PDF donné."""
    doc = pymupdf.open(str(pdf_path))
    extracted_pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        content = extract_page_content(page)
        if content:
            extracted_pages.append(
                {
                    "page": page_num + 1,
                    "text": content,
                }
            )
    doc.close()
    return extracted_pages


def recursive_split_text(
    text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Découpe un texte de manière récursif en respectant la hiérarchie des séparateurs :

    ['\n\n', '\n', '. ', '; ', ', ', ' ', '']
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Recherche du premier séparateur valide présent dans le texte
    chosen_sep = ""
    for sep in SEPARATORS:
        if sep == "":
            chosen_sep = ""
            break
        if sep in text:
            chosen_sep = sep
            break

    # Découpage du texte selon le séparateur choisi
    if chosen_sep != "":
        splits = text.split(chosen_sep)
    else:
        # Si aucun séparateur n'est trouvé, découpe par caractères
        splits = [
            text[i : i + chunk_size]
            for i in range(0, len(text), chunk_size - chunk_overlap)
        ]
        return [s.strip() for s in splits if s.strip()]

    # Regroupement des morceaux (splits) pour atteindre chunk_size sans dépasser
    chunks = []
    current_chunk = []
    current_length = 0

    for split in splits:
        split_len = len(split) + (len(chosen_sep) if current_chunk else 0)

        if current_length + split_len > chunk_size and current_chunk:
            combined = chosen_sep.join(current_chunk).strip()
            if combined:
                chunks.append(combined)

            # Gestion du recouvrement (overlap)
            overlap_length = 0
            new_current = []
            for piece in reversed(current_chunk):
                if overlap_length + len(piece) <= chunk_overlap:
                    new_current.insert(0, piece)
                    overlap_length += len(piece)
                else:
                    break
            current_chunk = new_current
            current_length = sum(len(p) for p in current_chunk) + max(
                0, len(current_chunk) - 1
            ) * len(chosen_sep)

        current_chunk.append(split)
        current_length += split_len

    if current_chunk:
        combined = chosen_sep.join(current_chunk).strip()
        if combined:
            chunks.append(combined)

    return chunks


def process_all_pdfs() -> list[dict]:
    """Traite tous les PDF de data/raw/ et génère la liste des chunks avec métadonnées."""
    all_chunks = []
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(
            f"Aucun PDF trouvé dans {RAW_DIR}. Placez des fichiers PDF dans data/raw/ avant de relancer."
        )
        return all_chunks

    for pdf_path in tqdm(pdf_files, desc="Parsing PyMuPDF & Découpage"):
        pages = extract_pdf_pages(pdf_path)
        for page_data in pages:
            page_num = page_data["page"]
            page_text = page_data["text"]

            sub_chunks = recursive_split_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
            for idx, chunk in enumerate(sub_chunks):
                all_chunks.append(
                    {
                        "chunk_id": f"{pdf_path.stem}_p{page_num}_c{idx+1}",
                        "source": pdf_path.name,
                        "page": page_num,
                        "text": chunk,
                        "char_count": len(chunk),
                    }
                )
    return all_chunks


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    chunks = process_all_pdfs()

    output_path = PROCESSED_DIR / "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] {len(chunks)} chunks récursifs générés et sauvegardés dans {output_path}")


if __name__ == "__main__":
    main()

