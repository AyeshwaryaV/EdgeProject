"""
train_and_export.py
Run this on your laptop to train K-Means and generate C++ arrays for ESP32.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("="*60)
print("STEP 1: Loading dataset...")
print("="*60)

# Load your dataset
df = pd.read_csv('engine_data.csv')

# Clean column names
df.columns = [c.strip() for c in df.columns]

# Define 6 features (no label)
feature_cols = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 
                'Coolant pressure', 'lub oil temp', 'Coolant temp']

# Check if all columns exist
missing = [col for col in feature_cols if col not in df.columns]
if missing:
    print(f"❌ ERROR: Missing columns: {missing}")
    print(f"   Available columns: {df.columns.tolist()}")
    exit()

print(f"✅ Loaded {len(df)} rows")
print(f"✅ Features: {feature_cols}")

# Filter ONLY normal data (Engine Condition == 0) for training
print("\n" + "="*60)
print("STEP 2: Training K-Means on NORMAL data only...")
print("="*60)

normal_data = df[df['Engine Condition'] == 0][feature_cols].copy()
print(f"✅ Normal samples for training: {len(normal_data)}")

# Remove missing values
normal_data = normal_data.dropna()
print(f"✅ Normal samples after cleaning: {len(normal_data)}")

# Z-score normalization
scaler = StandardScaler()
normalized_data = scaler.fit_transform(normal_data)

# Train K-Means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(normalized_data)
print(f"✅ K-Means training complete with {kmeans.n_clusters} clusters")

# Calculate anomaly threshold (95th percentile of distances)
print("\n" + "="*60)
print("STEP 3: Calculating anomaly threshold...")
print("="*60)

distances = []
for point in normalized_data:
    dist_to_centroids = np.linalg.norm(kmeans.cluster_centers_ - point, axis=1)
    distances.append(np.min(dist_to_centroids))

threshold = np.percentile(distances, 95)
print(f"✅ Anomaly threshold (95th percentile): {threshold:.6f}")

# ============================================
# PRINT C++ ARRAYS - COPY THESE INTO ESP32 CODE
# ============================================
print("\n" + "="*60)
print("📋 COPY THE FOLLOWING ARRAYS INTO esp32_inference.ino")
print("="*60)

print(f"\n// Number of features and clusters")
print(f"#define NUM_FEATURES {len(feature_cols)}")
print(f"#define NUM_CLUSTERS {kmeans.n_clusters}")

print(f"\n// Z-score scaling parameters (mean and std for each feature)")
print(f"const float SCALER_MEAN[{len(feature_cols)}] = {{ {', '.join([f'{x:.6f}f' for x in scaler.mean_])} }};")
print(f"const float SCALER_STD[{len(feature_cols)}] = {{ {', '.join([f'{x:.6f}f' for x in scaler.scale_])} }};")

print(f"\n// Cluster centroids (3 clusters × {len(feature_cols)} features)")
print(f"const float CENTROIDS[NUM_CLUSTERS][NUM_FEATURES] = {{")
for i, centroid in enumerate(kmeans.cluster_centers_):
    comma = "," if i < kmeans.n_clusters - 1 else ""
    print(f"    {{ {', '.join([f'{x:.6f}f' for x in centroid])} }}{comma}")
print(f"}};")

print(f"\n// Anomaly detection threshold")
print(f"#define ANOMALY_THRESHOLD {threshold:.6f}f")

print("\n" + "="*60)
print("✅ Training complete! Copy the arrays above into esp32_inference.ino")
print("="*60)

# Also save to JSON for reference
import json
config = {
    'feature_columns': feature_cols,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_std': scaler.scale_.tolist(),
    'centroids': kmeans.cluster_centers_.tolist(),
    'threshold': float(threshold),
    'num_clusters': kmeans.n_clusters,
    'num_features': len(feature_cols)
}
with open('model_config.json', 'w') as f:
    json.dump(config, f, indent=2)
print("\n📁 Config also saved to 'model_config.json'")
