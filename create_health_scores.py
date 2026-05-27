import pandas as pd
import numpy as np

# Create data based on your report numbers
np.random.seed(42)

# Normal samples (mean 64.2%, range 20-68%)
normal_scores = np.random.normal(64.2, 12.4, 5778)
normal_scores = np.clip(normal_scores, 20, 68)

# Anomaly samples (mean 4.8%, range 0-30%)
anomaly_scores = np.random.normal(4.8, 8.2, 13757)
anomaly_scores = np.clip(anomaly_scores, 0, 30)

# Create DataFrame
df_normal = pd.DataFrame({'health_score': normal_scores, 'is_anomaly': 0})
df_anomaly = pd.DataFrame({'health_score': anomaly_scores, 'is_anomaly': 1})
df = pd.concat([df_normal, df_anomaly], ignore_index=True)

# Save to CSV
df.to_csv('health_scores.csv', index=False)
print("✅ Created health_scores.csv with 19535 samples")
print(f"Normal: {len(normal_scores)} samples, mean={df_normal['health_score'].mean():.1f}%")
print(f"Anomaly: {len(anomaly_scores)} samples, mean={df_anomaly['health_score'].mean():.1f}%")