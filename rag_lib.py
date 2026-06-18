"""
rag_lib.py — Τα δομικά κομμάτια του Advanced RAG.

Χτίζει πάνω στο μάθημα 4: ChromaDB + SentenceTransformer('all-MiniLM-L6-v2')
+ ClaudeClient.ask(). Εδώ προσθέτουμε: chunking με overlap, query rewriting,
re-ranking με cross-encoder, config-driven retrieval και evaluation.
"""

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from ai_python_lab.claude_client import ClaudeClient

# --- Μοντέλα (φορτώνονται μία φορά) ---------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")              # ίδιο με μάθημα 4
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # ΝΕΟ

# Lazy: ο ClaudeClient (από το src του repo) φτιάχνεται μόνο όταν χρειαστεί,
# ώστε το ablation χωρίς rewrite/judge να τρέχει ακόμη και χωρίς API key.
_claude = None
def get_claude() -> ClaudeClient:
    global _claude
    if _claude is None:
        _claude = ClaudeClient()
    return _claude


# ===========================================================================
# 1) CHUNKING — overlap=0 σημαίνει «fixed» όπως στο μάθημα 4
# ===========================================================================
def fixed_chunks(text: str, size: int = 500) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


def overlap_chunks(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    """Με overlap, κάθε chunk «κουβαλάει» το τέλος του προηγούμενου."""
    step = max(1, size - overlap)
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


# ===========================================================================
# 2) STORE & RETRIEVE (ChromaDB, in-memory για το μάθημα)
# ===========================================================================
def build_collection(chunks: list[str], name: str = "docs"):
    client = chromadb.Client()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name)
    col.add(
        documents=chunks,
        embeddings=embedder.encode(chunks).tolist(),
        ids=[f"c_{i}" for i in range(len(chunks))],
    )
    return col


def retrieve(col, question: str, k: int = 3) -> list[str]:
    q_emb = embedder.encode([question]).tolist()
    res = col.query(query_embeddings=q_emb, n_results=k)
    return res["documents"][0]


# ===========================================================================
# 3) QUERY REWRITING
# ===========================================================================
def rewrite_query(question: str) -> str:
    system = (
        "Ξαναγράφεις ερωτήσεις σε καθαρές λέξεις-κλειδιά, κατάλληλες για "
        "αναζήτηση σε βάση εγγράφων (χωρίς φλυαρίες). Επίστρεφε ΜΟΝΟ τη νέα ερώτηση."
    )
    return get_claude().ask_with_system(system, question, max_tokens=60).strip()


# ===========================================================================
# 4) RE-RANKING — φέρε πολλά, ξαναβαθμολόγησε, κράτα λίγα
# ===========================================================================
def rerank(question: str, docs: list[str], top_n: int = 3) -> list[str]:
    pairs = [(question, d) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]


# ===========================================================================
# 5) PIPELINES — απλά (για τα μαθήματα 02-05)
# ===========================================================================
def naive_retrieval(col, question: str, k: int = 3) -> list[str]:
    return retrieve(col, question, k=k)


def advanced_retrieval(col, question: str, k_first: int = 10, top_n: int = 3) -> list[str]:
    better_q = rewrite_query(question)
    candidates = retrieve(col, better_q, k=k_first)
    return rerank(better_q, candidates, top_n=top_n)


def answer(question: str, context_chunks: list[str]) -> str:
    context = "\n---\n".join(context_chunks)
    return get_claude().ask(f"Με βάση το κείμενο:\n{context}\n\nΕρώτηση: {question}")


# ===========================================================================
# 6) CONFIG-DRIVEN retrieval — ΓΙΑ ΤΟ ΕΡΓΑΣΤΗΡΙΟ (ablation & challenge)
#    Ένας «διακόπτης» για κάθε τεχνική, ώστε να τις δοκιμάζουμε μία-μία.
# ===========================================================================
def configurable_retrieve(col, question, *, use_rewrite, k_first, top_n, use_rerank):
    q = rewrite_query(question) if use_rewrite else question
    k = k_first if use_rerank else top_n          # με rerank φέρνουμε περισσότερα
    docs = retrieve(col, q, k=k)
    if use_rerank:
        docs = rerank(q, docs, top_n=top_n)
    return docs


# ===========================================================================
# 7) EVALUATION — Hit Rate@k & MRR με keyword matching
# ===========================================================================
def keyword_score(retrieved: list[str], keyword: str) -> tuple[int, float]:
    for rank, doc in enumerate(retrieved, start=1):
        if keyword.lower() in doc.lower():
            return 1, 1.0 / rank
    return 0, 0.0


def evaluate(col, golden, retriever) -> tuple[float, float]:
    """retriever: callable(col, question) -> list[str]. Επιστρέφει (hit_rate, mrr)."""
    hits, rrs = [], []
    for question, keyword in golden:
        retrieved = retriever(col, question)
        h, r = keyword_score(retrieved, keyword)
        hits.append(h)
        rrs.append(r)
    return sum(hits) / len(hits), sum(rrs) / len(rrs)
