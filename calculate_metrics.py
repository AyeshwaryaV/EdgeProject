# calculate_metrics.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load your benchmark results (from run_realistic_benchmarks.py)
# Your results were:
results = {
    'K-Means': {'Accuracy': 0.6383, 'Precision': 0.6963, 'Recall': 0.7560, 'F1': 0.7249},
    'Random Forest': {'Accuracy': 0.9311, 'Precision': 0.9371, 'Recall': 0.9549, 'F1': 0.9459},
    'XGBoost': {'Accuracy': 0.9299, 'Precision': 0.9415, 'Recall': 0.9476, 'F1': 0.9446},
    'MLP NN': {'Accuracy': 0.8162, 'Precision': 0.7785, 'Recall': 0.9903, 'F1': 0.8717}
}

# Create comparison table
df_results = pd.DataFrame(results).T
print("\n=== MODEL PERFORMANCE COMPARISON ===")
print(df_results.to_string())

# Plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
x = np.arange(len(metrics))
width = 0.2
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

for i, (model, scores) in enumerate(results.items()):
    values = [scores[m] for m in metrics]
    ax.bar(x + i*width, values, width, label=model, color=colors[i])

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison for Engine Fault Detection')
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(metrics)
ax.legend(loc='lower right')
ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig('model_comparison_q1.png', dpi=300)
print("\n✅ Saved: model_comparison_q1.png")