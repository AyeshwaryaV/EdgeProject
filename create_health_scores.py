import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples from your test set
n_normal = 1443
n_anomaly = 2464

# Generate normal health scores (mean 64.2%, std 12.4%, range 20-68%)
normal_scores = np.random.normal(64.2, 12.4, n_normal)
normal_scores = np.clip(normal_scores, 20, 68)

# Generate anomaly health scores (mean 4.8%, std 8.2%, range 0-30%)
anomaly_scores = np.random.normal(4.8, 8.2, n_anomaly)
anomaly_scores = np.clip(anomaly_scores, 0, 30)

# Create DataFrames
df_normal = pd.DataFrame({
    'health_score': normal_scores,
    'is_anomaly': [0] * n_normal
})

df_anomaly = pd.DataFrame({
    'health_score': anomaly_scores,
    'is_anomaly': [1] * n_anomaly
})

# Combine
df = pd.concat([df_normal, df_anomaly], ignore_index=True)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv('health_scores.csv', index=False)

print(f"✅ Created health_scores.csv successfully!")
print(f"Total samples: {len(df)}")
print(f"Normal samples: {len(df[df['is_anomaly']==0])}")
print(f"Anomaly samples: {len(df[df['is_anomaly']==1])}")
print(f"Normal mean health score: {df[df['is_anomaly']==0]['health_score'].mean():.1f}%")
print(f"Anomaly mean health score: {df[df['is_anomaly']==1]['health_score'].mean():.1f}%")
print(f"File size should be approximately 150 KB")