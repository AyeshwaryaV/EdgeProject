# inspect_dataset.py
import pandas as pd
import numpy as np

print("="*60)
print("📊 STRUCTURAL DATASET CHARACTERISTIC INSPECTION")
print("="*60)

# Load data
df = pd.read_csv('engine_data.csv')
df.columns = [c.strip() for c in df.columns]

features = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']

# Compute class distributions
y = df['Engine Condition'].values
print(f"Total Database Rows: {len(y)}")
print(f"Healthy Baselines (0): {np.sum(y == 0)} ({np.sum(y == 0)/len(y):.2%})")
print(f"Anomalous Records (1): {np.sum(y == 1)} ({np.sum(y == 1)/len(y):.2%})\n")

print("📈 FEATURE DISTRIBUTION COMPARISONS (Mean ± Std Dev):")
print(f"{'Sensor Feature':<25} | {'Normal (Class 0)':<20} | {'Anomaly (Class 1)':<20}")
print("-" * 73)

for col in features:
    normal_mean = df[df['Engine Condition'] == 0][col].mean()
    normal_std = df[df['Engine Condition'] == 0][col].style if hasattr(df[df['Engine Condition'] == 0][col], 'style') else df[df['Engine Condition'] == 0][col].std()
    anomaly_mean = df[df['Engine Condition'] == 1][col].mean()
    anomaly_std = df[df['Engine Condition'] == 1][col].std()
    print(f"{col:<25} | {normal_mean:>8.2f} ± {normal_std:>5.2f} | {anomaly_mean:>8.2f} ± {anomaly_std:>5.2f}")

# Check for identical rows with conflicting labels
duplicate_features = df.duplicated(subset=features, keep=False)
conflicting_rows = df[duplicate_features].groupby(features).filter(lambda x: x['Engine Condition'].nunique() > 1)

print("\n⚠️ CONFLICTING DATA BOUNDARY ANALYSIS:")
print(f" ├─ Total rows with overlapping feature values: {len(df[duplicate_features])}")
print(f" └─ Exact rows with identical sensors but opposite labels: {len(conflicting_rows)}")

if len(conflicting_rows) > 0:
    print("    👉 Analysis: Identical sensor measurements are labeled as both Normal and Anomaly.")
    print("                 This overlap limits classification accuracy when using these features alone.")
else:
    print("    👉 Analysis: Sensor vectors maintain unique spatial coordinates.")

print("="*60)