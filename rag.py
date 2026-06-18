"""
ai_python_lab/rag.py — Οι RAG συναρτήσεις του Μαθήματος 4, μαζεμένες στη βιβλιοθήκη.

Είναι ΟΙ ΙΔΙΕΣ συναρτήσεις που γράψαμε στο μάθημα 4 — load_and_chunk,
load_and_chunk_pdf, store_chunks, ask_rag — εδώ σε επαναχρησιμοποιήσιμη μορφή,
ώστε να τις καλεί οποιοδήποτε script ή app (π.χ. το Streamlit app του μαθήματος 5).

Μία διαφορά από το μάθημα 4: αντί για global `model` / `collection`, τα περνάμε
ως παραμέτρους (καθαρές συναρτήσεις). Οι βοηθοί get_embedder() / get_collection()
τα φτιάχνουν μία φορά, ώστε να μη χρειάζεται να τα ξαναγράφει κανείς.

Χρειάζεται:  uv pip install chromadb sentence-transformers pymupdf
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from claude_client import ClaudeClient

EMBED_MODEL = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"


# ── Setup: φτιάχνονται ΜΙΑ φορά (cached) ────────────────────────────────────
@lru_cache(maxsize=None)
def get_embedder(name: str = EMBED_MODEL) -> SentenceTransformer:
    """Το μοντέλο embeddings του μαθήματος 4 (φορτώνεται μία φορά)."""
    return SentenceTransformer(name)


@lru_cache(maxsize=None)
def get_collection(db_path: str = "./my_db", name: str = "documents"):
    """Ανοίγει (ή φτιάχνει) μια ChromaDB collection αποθηκευμένη στον δίσκο."""
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=name)


# ── Βήμα 1 + 2: Φόρτωση & chunking  (μάθημα 4) ──────────────────────────────
def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    """Σπάει κείμενο σε chunks των ~chunk_size χαρακτήρων (αγνοεί τα κενά)."""
    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
        if text[i:i + chunk_size].strip()
    ]


def load_and_chunk(filepath: str, chunk_size: int = 500) -> list[str]:
    """Διαβάζει ένα .txt και το σπάει σε chunks.  (μάθημα 4 · 01_txt_loader)"""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return chunk_text(text, chunk_size)


def load_and_chunk_pdf(filepath: str, chunk_size: int = 500) -> list[str]:
    """Διαβάζει ένα PDF (PyMuPDF) και το σπάει σε chunks.  (μάθημα 4 · 02_pdf_loader)"""
    import fitz  # PyMuPDF — lazy import, ώστε για .txt να μη χρειάζεται
    doc = fitz.open(filepath)
    text = "".join(page.get_text() for page in doc)
    return chunk_text(text, chunk_size)


# ── Βήμα 3: Αποθήκευση στο ChromaDB  (μάθημα 4 · 05_chatbot, με dedup) ───────
def store_chunks(collection, chunks: list[str], source_name: str, model=None) -> int:
    """
    Αποθηκεύει chunks ως embeddings στο ChromaDB. Παραλείπει όσα υπάρχουν ήδη.
    Επιστρέφει πόσα ΝΕΑ προστέθηκαν.
    """
    model = model or get_embedder()
    ids = [f"{source_name}_chunk_{i:04d}" for i in range(len(chunks))]
    existing = set(collection.get(ids=ids, include=[]).get("ids", []))

    new_ids, new_docs, new_embs = [], [], []
    for cid, chunk in zip(ids, chunks):
        if cid in existing:
            continue
        new_ids.append(cid)
        new_docs.append(chunk)
        new_embs.append(model.encode(chunk).tolist())

    if new_ids:
        collection.add(documents=new_docs, embeddings=new_embs, ids=new_ids)
    return len(new_ids)


# ── Βήμα 4: Ερώτηση → Εύρεση → Απάντηση  (μάθημα 4 · 03_rag_query / 05_chatbot)
def retrieve(question: str, collection, n_results: int = 3, model=None) -> list[str]:
    """Βρίσκει τα n πιο σχετικά chunks για την ερώτηση."""
    model = model or get_embedder()
    q_emb = model.encode([question]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=n_results)
    return res["documents"][0] if res["documents"] else []


def build_prompt(question: str, context_chunks: list[str]) -> str:
    """Το prompt του μαθήματος 4: «Με βάση το κείμενο ... Ερώτηση: ...»."""
    context = "\n---\n".join(context_chunks)
    return (
        "Απάντησε στην ερώτηση ΜΟΝΟ με βάση το παρακάτω κείμενο. "
        "Αν η απάντηση δεν υπάρχει στο κείμενο, πες το ξεκάθαρα.\n\n"
        f"Κείμενο:\n{context}\n\nΕρώτηση: {question}"
    )


def answer_with_context(question, context_chunks, client=None, max_tokens=1000) -> str:
    """Μη-streaming απάντηση, με έτοιμα chunks ως context."""
    client = client or ClaudeClient()
    return client.ask(build_prompt(question, context_chunks), max_tokens=max_tokens)


def stream_with_context(question, context_chunks, client=None, max_tokens=1000):
    """
    Streaming απάντηση (generator), με έτοιμα chunks ως context.
    Κάνει `yield` κάθε κομμάτι κειμένου — όπως το streaming του μαθήματος 3.
    """
    client = client or ClaudeClient()  # ΕΙΝΑΙ Anthropic client → έχει .messages.stream
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": build_prompt(question, context_chunks)}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def ask_rag(question: str, collection, n_results: int = 3, model=None, client=None,
            max_tokens: int = 1000) -> str:
    """
    Η «μία κλήση» του μαθήματος 4: εύρεση σχετικών chunks + απάντηση από το Claude.
    (μάθημα 4 · 03_rag_query / 05_chatbot)
    """
    chunks = retrieve(question, collection, n_results=n_results, model=model)
    return answer_with_context(question, chunks, client=client, max_tokens=max_tokens)
