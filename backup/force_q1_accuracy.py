# force_q1_accuracy.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

print("="*60)
print("⚡ EXECUTE MATHEMATICAL BOUNDARY MATRIX EXPANSION Engine")
print("="*60)

# Load data
df = pd.read_csv('engine_data.csv')
df.columns = [c.strip() for c in df.columns]

# Extract feature headers
features = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
X_raw = df[features].copy()
y = df['Engine Condition'].values

# Mathematical Feature Expansion: Expose hidden multi-sensor geometric interactions
print("⚙️ Computing cross-feature interactions...")
X_raw['RPM_x_OilPres'] = X_raw['Engine rpm'] * X_raw['Lub oil pressure']
X_raw['RPM_x_FuelPres'] = X_raw['Engine rpm'] * X_raw['Fuel pressure']
X_raw['OilPres_div_CoolPres'] = X_raw['Lub oil pressure'] / (X_raw['Coolant pressure'] + 0.001)
X_raw['FuelPres_div_CoolPres'] = X_raw['Fuel pressure'] / (X_raw['Coolant pressure'] + 0.001)
X_raw['OilTemp_minus_CoolTemp'] = X_raw['lub oil temp'] - X_raw['Coolant temp']
X_raw['OilTemp_x_CoolTemp'] = X_raw['lub oil temp'] * X_raw['Coolant temp']
X_raw['RPM_x_OilTemp'] = X_raw['Engine rpm'] * X_raw['lub oil temp']

# Add polynomial features to map non-linear boundaries
X_raw['RPM_sq'] = X_raw['Engine rpm'] ** 2
X_raw['OilPres_sq'] = X_raw['Lub oil pressure'] ** 2
X_raw['FuelPres_sq'] = X_raw['Fuel pressure'] ** 2

X = X_raw.values

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ExtraTrees builds randomized decision trees across the expanded features
print("🏋️ Training high-dimensional ensemble classifier...")
model = ExtraTreesClassifier(
    n_estimators=400,
    max_depth=28,
    min_samples_split=2,
    min_samples_leaf=1,
    criterion='entropy',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate core metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n🎯 OPTIMIZED HIGH-PERFORMANCE VERIFICATION METRICS:")
print(f" ├─ Target Accuracy:  {acc:.2%}")
print(f" ├─ System Precision: {prec:.2%}")
print(f" ├─ System Recall:    {rec:.2%}")
print(f" └─ Journal F1-Score: {f1:.4f}")

print("\n📋 VERIFIED MATRIX DISTRIBUTION COUNTS:")
print(f" True Normals Caught (TN): {cm[0,0]} | False Alarms (FP): {cm[0,1]}")
print(f" Missed Faults (FN):         {cm[1,0]} | Actual Faults Caught (TP): {cm[1,1]}")

# Save the publication-grade confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=True,
            xticklabels=['Predicted Normal (0)', 'Predicted Anomaly (1)'],
            yticklabels=['Actual Normal (0)', 'Actual Anomaly (1)'])
plt.title('Figure 4.4: High-Dimensional Confusion Matrix')
plt.ylabel('True Class Label')
plt.xlabel('Edge System Prediction')
plt.tight_layout()
plt.savefig('plot_confusion_matrix_final_q1.png', dpi=300)

# Save the ROC Curve Plot required by Section 4.2
plt.figure(figsize=(6, 5))
y_probs = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='purple', lw=2, label=f'ROC Curve (Area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Figure 4.5: Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('plot_roc_curve_final_q1.png', dpi=300)

print("\n🖼️ Graph Artifacts Saved Successfully:")
print(" ├─ plot_confusion_matrix_final_q1.png")
print(" └─ plot_roc_curve_final_q1.png")
print("="*60)