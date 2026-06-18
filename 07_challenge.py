"""
07_challenge.py — ΤΟ CHALLENGE: νίκησε το baseline!

Το naive RAG πιάνει MRR γύρω στο 0.60 σε αυτό το corpus. Στόχος σου:
να φτάσεις ή να ξεπεράσεις το TARGET παρακάτω, ρυθμίζοντας τους «διακόπτες».

Κανόνες:
  • Άλλαξε όποιες ρυθμίσεις θες (chunking, rewrite, rerank, top_n...).
  • Ξανατρέξε. Το καλύτερό σου σκορ αποθηκεύεται στο best_score.txt.
  • Προσπάθησε να καταλάβεις ΓΙΑΤΙ κάθε αλλαγή ανεβάζει ή ρίχνει το σκορ.

Δες αν θα φτάσεις πάνω από το TARGET — και μετά αν θα πιάσεις τέλειο 1.00!
"""

import os
from rag_lib import (overlap_chunks, build_collection,
                     configurable_retrieve, evaluate)
from sample_data import SAMPLE_TEXT, GOLDEN

TARGET = 0.90          # 🎯 ο στόχος σου σε MRR
BEST_FILE = "best_score.txt"

# ===========================================================================
#  👇  ΟΙ ΡΥΘΜΙΣΕΙΣ ΣΟΥ — πείραξέ τες για να νικήσεις το TARGET
# ===========================================================================
CHUNK_SIZE  = 200
OVERLAP     = 0
USE_REWRITE = False
USE_RERANK  = False
K_FIRST     = 8
TOP_N       = 3
# ===========================================================================


def load_best() -> float:
    if os.path.exists(BEST_FILE):
        return float(open(BEST_FILE).read().strip() or 0)
    return 0.0


def save_best(score: float):
    open(BEST_FILE, "w").write(f"{score:.4f}")


def main():
    chunks = overlap_chunks(SAMPLE_TEXT, size=CHUNK_SIZE, overlap=OVERLAP)
    col = build_collection(chunks)
    retriever = lambda c, q: configurable_retrieve(
        c, q, use_rewrite=USE_REWRITE, k_first=K_FIRST,
        top_n=TOP_N, use_rerank=USE_RERANK)

    hit, mrr = evaluate(col, GOLDEN, retriever)
    best = load_best()

    print(f"\n  Hit Rate: {hit:.2f}   |   MRR: {mrr:.2f}   |   🎯 TARGET: {TARGET:.2f}")

    if mrr >= TARGET:
        print("  ✅ Νίκησες το TARGET! Μπράβο!")
    else:
        print(f"  ⏳ Ακόμη {TARGET - mrr:.2f} για το TARGET — δοκίμασε άλλες ρυθμίσεις.")

    if mrr > best:
        save_best(mrr)
        print(f"  🏆 ΝΕΟ ρεκόρ! (προηγούμενο: {best:.2f})\n")
    else:
        print(f"  (ρεκόρ μέχρι τώρα: {best:.2f})\n")


if __name__ == "__main__":
    main()
