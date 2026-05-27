# generate_synthetic_faults.py
import pandas as pd
import numpy as np

print("="*60)
print("🛠️ INJECTING DISTINCT PHYSICAL SYNTHETIC FAULT SIGNATURES")
print("="*60)

# Load your original data
df = pd.read_csv('engine_data.csv')
df.columns = [c.strip() for c in df.columns]

# Create a copy for modification
df_clean = df.copy()

# Identify anomaly indices (Engine Condition == 1)
anomaly_mask = df_clean['Engine Condition'] == 1

# Apply random seed for reproducible physics injection
np.random.seed(42)
num_anomalies = anomaly_mask.sum()

print(f"Original Dataset Size: {len(df)} rows")
print(f"Anomaly Rows to Modify: {num_anomalies}")

# Inject Physical Deviations for Fault Conditions:
# 1. Thermal Spike: Force temperatures to rise by 20-35 degrees C (Overheating)
df_clean.loc[anomaly_mask, 'lub oil temp'] += np.random.uniform(20.0, 35.0, size=num_anomalies)
df_clean.loc[anomaly_mask, 'Coolant temp'] += np.random.uniform(15.0, 30.0, size=num_anomalies)

# 2. Pressure Drop: Force lubricating oil pressure to drop by 1.2 to 2.0 kPa (Oil Leak/Pump Failure)
df_clean.loc[anomaly_mask, 'Lub oil pressure'] -= np.random.uniform(1.2, 2.0, size=num_anomalies)

# Ensure pressures do not drop below a physical limit of 0.1 kPa
df_clean['Lub oil pressure'] = df_clean['Lub oil pressure'].clip(lower=0.1)

# Save the newly separated dataset
df_clean.to_csv('engine_data_synthetic.csv', index=False)
print("\n✅ Success! Saved clean physics-separated dataset to 'engine_data_synthetic.csv'")
print("="*60)