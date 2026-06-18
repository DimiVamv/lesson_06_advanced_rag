"""
06_experiment.py

Στόχος: Να ερευνήσουμε ποια τεχνική βοηθάει και πόσο.

Πώς δουλεύει ο κώδικας:
  1) Τρέχουμε το script όπως είναι (baseline με όλες τις τεχνικές κλειστές).
  2) Σημειώνουμε Hit Rate & MRR στον πίνακα της διαφάνειας.
  3) Ενεργοποιούμε μία τεχνική τη φορά (βάζουμε True), τρέχουμε και σημειώνουμε εκ νέου.
  4) Στο τέλος ενεργοποιούμε όλες τις τεχνικές. Ποιος συνδυασμός είναι καλύτερος;

Μία ρύθμιση πρέπει να αλλάζει κάθε φορά, αλλιώς δεν θα ξέρουμε τι κάνει τη διαφορά.
"""

from rag_lib import (overlap_chunks, build_collection,
                     configurable_retrieve, evaluate)
from sample_data import SAMPLE_TEXT, GOLDEN

# ===========================================================================
#  👇  ΟΙ ΡΥΘΜΙΣΕΙΣ ΣΟΥ — άλλαξε εδώ και ξανατρέξε
# ===========================================================================
CHUNK_SIZE  = 200      # μέγεθος κάθε chunk (χαρακτήρες)
OVERLAP     = 50        # επικάλυψη μεταξύ chunks (0 = fixed, δοκίμασε 60)
USE_REWRITE = True    # query rewriting με το LLM;  (χρειάζεται API key)
USE_RERANK  = True    # re-ranking με cross-encoder;
K_FIRST     = 5        # πόσα chunks φέρνουμε πριν το rerank
TOP_N       = 2        # πόσα κρατάμε τελικά ως context
# ===========================================================================


def main():
    chunks = overlap_chunks(SAMPLE_TEXT, size=CHUNK_SIZE, overlap=OVERLAP)
    col = build_collection(chunks)

    retriever = lambda c, q: configurable_retrieve(
        c, q, use_rewrite=USE_REWRITE, k_first=K_FIRST,
        top_n=TOP_N, use_rerank=USE_RERANK)

    hit, mrr = evaluate(col, GOLDEN, retriever)

    import csv
    from datetime import datetime

    # Αποθήκευση πειράματος σε CSV
    log_file = "experiments.csv"
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Γράψε headers την πρώτη φορά
        if f.tell() == 0:
            writer.writerow(["timestamp", "chunk_size", "overlap", "rewrite", "rerank", "k_first", "top_n", "hit_rate", "mrr"])
        writer.writerow([
            datetime.now().isoformat(),
            CHUNK_SIZE, OVERLAP, USE_REWRITE, USE_RERANK, K_FIRST, TOP_N,
            f"{hit:.4f}", f"{mrr:.4f}"
        ])

    print("─" * 44)
    print(f"  chunks={len(chunks)}  size={CHUNK_SIZE}  overlap={OVERLAP}")
    print(f"  rewrite={USE_REWRITE}  rerank={USE_RERANK}  top_n={TOP_N}")
    print("─" * 44)
    print(f"  Hit Rate : {hit:.2f}")
    print(f"  MRR      : {mrr:.2f}")
    print("─" * 44)


if __name__ == "__main__":
    main()
