# generate_q1_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier

print("="*60)
print("⚡ INITIALIZING ADVANCED GRADIENT-BOOSTING DIAGNOSTIC TRACE")
print("="*60)

# Load data
df = pd.read_csv('engine_data.csv')
df.columns = [c.strip() for c in df.columns]

# Extract feature headers
features = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
X_raw = df[features].copy()
y = df['Engine Condition'].values

# Advanced Engineering: Compute structural thermodynamic dependencies
X_raw['Oil_Coolant_Temp_Delta'] = X_raw['lub oil temp'] - X_raw['Coolant temp']
X_raw['Mechanical_Stress_Index'] = X_raw['Engine rpm'] * X_raw['Lub oil pressure']
X_raw['Hydraulic_Ratio'] = X_raw['Fuel pressure'] / (X_raw['Coolant pressure'] + 0.1)

X = X_raw.values

# Stratified partition matrix split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Configure XGBoost to map complex non-linear boundary constraints
xgb_model = XGBClassifier(
    n_estimators=350,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    scale_pos_weight=0.6,  # Strategically counter balance overlapping false alerts
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

# Calculate core metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n🎯 NEW OPTIMIZED HIGH-PERFORMANCE JOURNAL METRICS:")
print(f" ├─ Verification Accuracy: {acc:.2%}")
print(f" ├─ Precision Performance: {prec:.2%}")
print(f" ├─ Target Recall Rate:    {rec:.2%}")
print(f" └─ Calculated F1-Score:   {f1:.4f}")

print("\n📋 EXPANDED TEST CONFUSION MATRIX MAP:")
print(f" Actual Normal Caught (TN): {cm[0,0]} | False Alarms (FP): {cm[0,1]}")
print(f" Missed Faults (FN):         {cm[1,0]} | Actual Faults Caught (TP): {cm[1,1]}")

# Save the polished confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=True,
            xticklabels=['Predicted Normal (0)', 'Predicted Anomaly (1)'],
            yticklabels=['Actual Normal (0)', 'Actual Anomaly (1)'])
plt.title('Figure 4.4: High-Performance Boosted Confusion Matrix')
plt.ylabel('True Target Class Labels')
plt.xlabel('Edge System Evaluation Predictions')
plt.tight_layout()
plt.savefig('plot_confusion_matrix_final.png', dpi=300)

# Save the necessary ROC Curve Plot required by Section 4.2
plt.figure(figsize=(6, 5))
y_probs = xgb_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (Area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Figure 4.5: Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('plot_roc_curve_final.png', dpi=300)

print("\n🖼️ Graph Artifacts Saved Successfully:")
print(" ├─ plot_confusion_matrix_final.png")
print(" └─ plot_roc_curve_final.png")
print("="*60)