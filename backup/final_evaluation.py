# final_evaluation.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("="*60)
print("📊 M.TECH THESIS METRICS GENERATOR (SECTION 4.2 & 4.3)")
print("="*60)

# 1. Load your actual dataset
try:
    df = pd.read_csv('engine_data.csv')
    # Clean up column names to remove any accidental trailing spaces
    df.columns = [c.strip() for c in df.columns]
    print("✅ Successfully loaded engine_data.csv")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("Ensure final_evaluation.py is saved directly in C:\\Users\\Aishwarya\\OneDrive\\Desktop\\EdgeProject")
    exit()

# 2. Match your exact dataset column names
# Columns in your dataset match these patterns:
feature_cols = ['engine_rpm', 'lub_oil_pressure', 'fuel_pressure', 'coolant_pressure', 'lub_oil_temp', 'coolant_temp']

# Fallback column search if capitalization differs
if not all(col in df.columns for col in feature_cols):
    # Try case-insensitive mapping
    mapping = {col.lower().replace(" ", "_"): col for col in df.columns}
    feature_cols = [mapping.get(c.lower().replace(" ", "_"), c) for c in feature_cols]

X_raw = df[feature_cols].values

# Map ground truth column names
target_col = 'Engine Condition' if 'Engine Condition' in df.columns else 'Class'
y_true = df[target_col].values

# 3. Your exact ESP32 model constants (Matches your firmware parameters)
SCALER_MEAN = np.array([884.995012, 3.222497, 6.236310, 2.367934, 78.023934, 78.803030])
SCALER_STD = np.array([271.703991, 1.010284, 2.681359, 1.087083, 3.231583, 5.967972])
CENTROIDS = np.array([
    [0.086543, -0.031571, 0.027488, 0.007288, 1.903871, 0.092997],
    [0.096429, 0.980942, 0.119297, -0.048680, -0.380763, -0.144918],
    [-0.103787, -0.718424, -0.098960, 0.033535, -0.420935, 0.073457]
])
ANOMALY_THRESHOLD = 3.447889

# 4. Replicate the precise hardware math
X_scaled = (X_raw - SCALER_MEAN) / SCALER_STD

y_pred = []
for point in X_scaled:
    dists = np.linalg.norm(CENTROIDS - point, axis=1)
    min_dist = np.min(dists)
    # Flag as anomaly if it exceeds threshold bounds
    is_anomaly = 1 if min_dist > ANOMALY_THRESHOLD else 0
    y_pred.append(is_anomaly)

y_pred = np.array(y_pred)

# 5. Compute performance statistics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n📈 CORE CLASSIFICATION METRICS:")
print(f" ├─ Accuracy:  {acc:.2%}")
print(f" ├─ Precision: {prec:.2%}")
print(f" ├─ Recall:    {rec:.2%}")
print(f" └─ F1-Score:  {f1:.4f}")

# 6. Generate and save the Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_true, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Predicted Normal (0)', 'Predicted Anomaly (1)'],
            yticklabels=['Actual Normal (0)', 'Actual Anomaly (1)'])

plt.title('Figure 4.4: Embedded Anomaly Detection Confusion Matrix')
plt.ylabel('True Fleet Label Class')
plt.xlabel('Edge Predicted Classification')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("\n🖼️ Visual matrix saved successfully as 'confusion_matrix.png'")
print("="*60)