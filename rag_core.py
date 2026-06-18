"""
rag_core.py — Συνδέει το Streamlit app με τη βιβλιοθήκη μας.

Δεν ξαναγράφουμε RAG εδώ! Όλες οι συναρτήσεις (chunking, αποθήκευση, εύρεση,
απάντηση) είναι από το μάθημα 4. Τις έχουμε πακετάρει στη βιβλιοθήκη μας, στο
'ai_python_lab.rag'.

Εδώ κάνουμε μόνο 2 πράγματα ειδικά για το app
    α) φτιάχνουμε μία φορά το μοντέλο και τη συλλογή (collection) τοπικά για αυτό το app,
    β) προσθέτουμε ένα μικρό μετατροπέα/wrapper για αρχεία που ανεβαίνουν από τον browser (bytes).

Χρειάζεται uv pip install streamlit chromadb sentence-transformers pymupdf
"""

from pathlib import Path
import fitz  # PyMuPDF
import rag # η βιβλιοθήκη μας (οι συναρτήσεις του μαθήματος 4)

DB_PATH = str(Path(__file__).parent / "chroma_db")

# Τα ίδια "model" και "collection" του μαθήματος 4 — μέσω της βιβλιοθήκης.
_model = rag.get_embedder()
_collection = rag.get_collection(DB_PATH)


def load_and_chunk_upload(file_bytes: bytes, filename: str, chunk_size: int = 500) -> list[str]:
    """
    Το Streamlit δίνει το αρχείο ως bytes (όχι path, όπως στο μάθημα 4). Το
    διαβάζουμε εδώ και μετά καλούμε το ΙΔΙΟ rag.chunk_text() της βιβλιοθήκης.
    """
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
    else:
        text = file_bytes.decode("utf-8", errors="ignore")
    return rag.chunk_text(text, chunk_size)  # ← ίδιο chunking με το μάθημα 4


def index_chunks(chunks: list[str], source_name: str) -> int:
    """Αποθήκευση στο ChromaDB — rag.store_chunks() του μαθήματος 4."""
    return rag.store_chunks(_collection, chunks, source_name, model=_model)


def has_documents() -> bool:
    """True αν υπάρχει έστω ένα chunk αποθηκευμένο."""
    return _collection.count() > 0


def retrieve(question: str, k: int = 3) -> list[str]:
    """Εύρεση σχετικών chunks — rag.retrieve() του μαθήματος 4."""
    return rag.retrieve(question, _collection, n_results=k, model=_model)


def stream_answer(question: str, context_chunks: list[str]):
    """Streaming απάντηση — rag.stream_with_context() (μάθημα 3: streaming + μάθημα 4: RAG)."""
    return rag.stream_with_context(question, context_chunks)
