import matplotlib.pyplot as plt
import numpy as np

models = ['K-Means', 'Random Forest', 'XGBoost', 'MLP NN']
accuracy = [63.83, 93.11, 92.99, 81.62]
precision = [69.63, 93.71, 94.15, 77.85]
recall = [75.60, 95.49, 94.76, 99.03]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, accuracy, width, label='Accuracy')
ax.bar(x, precision, width, label='Precision')
ax.bar(x + width, recall, width, label='Recall')

ax.set_xlabel('Models')
ax.set_ylabel('Percentage (%)')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('model_performance_chart.png', dpi=300)
print("✅ Saved: model_performance_chart.png")