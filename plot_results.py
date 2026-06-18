"""
plot_results.py – Διάβασε το experiments.csv και φτιάξε ένα bar chart.
"""

import csv
import os
import matplotlib.pyplot as plt

csv_file = "experiments.csv"
if not os.path.exists(csv_file):
    print("Δεν βρέθηκε το experiments.csv. Τρέξε πρώτα το 06_experiment.py.")
    exit(1)

labels, scores = [], []
with open(csv_file, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = f"O={row['overlap']}" + ("+RW" if row['rewrite'] == 'True' else "") + ("+RR" if row['rerank'] == 'True' else "")
        labels.append(label)
        scores.append(float(row['mrr']))

plt.figure(figsize=(10, 6))
bars = plt.bar(labels, scores, color='lightgreen')
plt.ylim(0, 1.0)
plt.ylabel('MRR')
plt.title('Αποτελέσματα RAG πειραμάτων')
plt.axhline(y=0.90, color='red', linestyle='--', label='Target (0.90)')
plt.legend()
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{score:.2f}', ha='center', va='bottom')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('rag_results.png')
print("📊 Γράφημα αποθηκεύτηκε ως rag_results.png")
plt.show()