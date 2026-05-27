# optimize_rf_q1.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("="*60)
print("⚡ RUNNING HIGH-ACCURACY Q1 OPTIMIZATION PIPELINE")
print("="*60)

# Load data
df = pd.read_csv('engine_data.csv')
df.columns = [c.strip() for c in df.columns]

# Core features
features = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
X_raw = df[features].copy()
y = df['Engine Condition'].values

# Feature Engineering: Add multivariate physical interaction constraints
X_raw['Pressure_Ratio'] = X_raw['Lub oil pressure'] / (X_raw['Coolant pressure'] + 0.01)
X_raw['Temp_Diff'] = X_raw['lub oil temp'] - X_raw['Coolant temp']
X_raw['Thermal_Load'] = X_raw['Engine rpm'] * X_raw['lub oil temp']

X = X_raw.values

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Deep Forest Configuration to eliminate cross-over false alarms
rf_optimized = RandomForestClassifier(
    n_estimators=250, 
    max_depth=18,
    min_samples_split=4,
    class_weight={0: 1.0, 1: 1.3}, # Adjust weight to penalize false alarms
    random_state=42, 
    n_jobs=-1
)
rf_optimized.fit(X_train, y_train)

# Inference
y_pred = rf_optimized.predict(X_test)

# Metrics calculation
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n📈 CORE VERIFICATION METRICS EXPORT:")
print(f" ├─ Target Accuracy:  {acc:.2%}")
print(f" ├─ System Precision: {prec:.2%}")
print(f" ├─ System Recall:    {rec:.2%}")
print(f" └─ Journal F1-Score: {f1:.4f}")

# Save crisp visual for your draft chapter
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Predicted Normal (0)', 'Predicted Anomaly (1)'],
            yticklabels=['Actual Normal (0)', 'Actual Anomaly (1)'])
plt.title('Figure 4.4: Optimized Random Forest Confusion Matrix')
plt.ylabel('True Class Label')
plt.xlabel('Edge System Prediction')
plt.tight_layout()
plt.savefig('rf_confusion_matrix_optimized.png', dpi=300)
print("\n🖼️ Verification plot successfully written to 'rf_confusion_matrix_optimized.png'")
print("="*60)