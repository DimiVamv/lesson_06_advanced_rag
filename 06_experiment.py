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
OVERLAP     = 0        # επικάλυψη μεταξύ chunks (0 = fixed, δοκίμασε 60)
USE_REWRITE = False    # query rewriting με το LLM;  (χρειάζεται API key)
USE_RERANK  = False    # re-ranking με cross-encoder;
K_FIRST     = 8        # πόσα chunks φέρνουμε πριν το rerank
TOP_N       = 3        # πόσα κρατάμε τελικά ως context
# ===========================================================================


def main():
    chunks = overlap_chunks(SAMPLE_TEXT, size=CHUNK_SIZE, overlap=OVERLAP)
    col = build_collection(chunks)

    retriever = lambda c, q: configurable_retrieve(
        c, q, use_rewrite=USE_REWRITE, k_first=K_FIRST,
        top_n=TOP_N, use_rerank=USE_RERANK)

    hit, mrr = evaluate(col, GOLDEN, retriever)

    print("─" * 44)
    print(f"  chunks={len(chunks)}  size={CHUNK_SIZE}  overlap={OVERLAP}")
    print(f"  rewrite={USE_REWRITE}  rerank={USE_RERANK}  top_n={TOP_N}")
    print("─" * 44)
    print(f"  Hit Rate : {hit:.2f}")
    print(f"  MRR      : {mrr:.2f}")
    print("─" * 44)


if __name__ == "__main__":
    main()
