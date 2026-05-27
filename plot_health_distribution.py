# plot_health_distribution.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('health_scores.csv')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Boxplot
sns.boxplot(x='is_anomaly', y='health_score', data=df, ax=axes[0])
axes[0].set_xlabel('Engine State (0=Normal, 1=Anomaly)')
axes[0].set_ylabel('Health Score (%)')
axes[0].set_title('Health Score Distribution: Normal vs Anomaly')
axes[0].set_xticklabels(['NORMAL', 'ANOMALY'])

# Violin plot
sns.violinplot(x='is_anomaly', y='health_score', data=df, ax=axes[1])
axes[1].set_xlabel('Engine State (0=Normal, 1=Anomaly)')
axes[1].set_ylabel('Health Score (%)')
axes[1].set_title('Health Score Violin Plot')
axes[1].set_xticklabels(['NORMAL', 'ANOMALY'])

plt.tight_layout()
plt.savefig('health_score_distribution.png', dpi=300)
print("✅ Saved: health_score_distribution.png")