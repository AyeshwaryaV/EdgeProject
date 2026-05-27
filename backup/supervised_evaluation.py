import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

print("="*60)
print("🚀 TRAINING SUPERVISED VEHICLE DIAGNOSTIC MODEL")
print("="*60)

# Load data
df = pd.read_csv('engine_data.csv')
df.columns = [c.strip() for c in df.columns]

features = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
X = df[features].values
y = df['Engine Condition'].values

# Stratified split to keep the normal/anomaly ratio identical in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Random Forest with balanced class weights
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# Run Inference
y_pred = rf.predict(X_test_scaled)

# Compute Core Verification Scores
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n📈 JOURNAL PERFORMANCE RESULTS:")
print(f" ├─ Verification Accuracy: {acc:.2%}")
print(f" ├─ Precision Accuracy:    {prec:.2%}")
print(f" ├─ True Recall Rate:      {rec:.2%}")
print(f" └─ Optimized F1-Score:    {f1:.4f}")

# Print text matrix directly to console
print("\n📋 TEST MATRIX DISTRIBUTION COUNTS:")
print(f" True Normals Caught:   {cm[0,0]} | False Alarms Generated: {cm[0,1]}")
print(f" Missed Fault Vectors:  {cm[1,0]} | True Faults Caught:     {cm[1,1]}")

# Generate Publication Heatmap Image
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=True,
            xticklabels=['Predicted Normal (0)', 'Predicted Anomaly (1)'],
            yticklabels=['Actual Normal (0)', 'Actual Anomaly (1)'])
plt.title('Figure 4.4: Supervised Random Forest Confusion Matrix')
plt.ylabel('True Target Class')
plt.xlabel('Edge System Class Prediction')
plt.tight_layout()
plt.savefig('rf_confusion_matrix_q1.png', dpi=300)
print("\n🖼️ Graph Saved: 'rf_confusion_matrix_q1.png' has been written to your folder.")
print("="*60)