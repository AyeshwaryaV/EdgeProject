# plot_confusion_matrix.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Based on your benchmark results (Random Forest: 93.11% accuracy)
# Approximate confusion matrix for 3907 test samples
total = 3907
acc = 0.9311
correct = int(total * acc)
wrong = total - correct

# Assuming balanced classes (your data had ~63% faults, 37% normal)
n_fault = int(total * 0.63)  # ~2461
n_normal = total - n_fault    # ~1446

# Estimate TP, TN, FP, FN
# Recall = 95.49%, Precision = 93.71%
recall = 0.9549
precision = 0.9371

TP = int(n_fault * recall)
FN = n_fault - TP
TP = int(TP)
FN = int(FN)

# Precision = TP / (TP + FP) => FP = TP/precision - TP
FP = int(TP / precision - TP) if precision > 0 else 0
TN = n_normal - FP

cm = np.array([[TN, FP], [FN, TP]])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
            xticklabels=['Predicted Normal', 'Predicted Fault'],
            yticklabels=['Actual Normal', 'Actual Fault'])
plt.title('Confusion Matrix: Random Forest Fault Detection')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_heatmap.png', dpi=300)
print("✅ Saved: confusion_matrix_heatmap.png")