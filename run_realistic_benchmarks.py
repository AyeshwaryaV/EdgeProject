# run_realistic_benchmarks.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

print("="*80)
print("📊 EXECUTING SUPERVISED RANDOM FOREST BENCHMARKING PIPELINE")
print("="*80)

# Load the realistic dataset
try:
    df = pd.read_csv('engine_data_realistic.csv')
except FileNotFoundError:
    print("❌ Error: 'engine_data_realistic.csv' not found. Run generate_realistic_data.py first.")
    exit()

features = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
X = df[features].values
y = df['Engine Condition'].values

# Stratified split to ensure identical class distributions across train/test horizons
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"📦 Training Matrix Vectors: {X_train.shape[0]}")
print(f"📦 Evaluation Test Vectors: {X_test.shape[0]}")
print("🏋️ Training comparative models across feature planes...")

# 1. Random Forest (Supervised comparator)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 2. XGBoost (Supervised comparator)
xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# Compute Core Metrics
models = {
    'Random Forest': y_pred_rf,
    'XGBoost': y_pred_xgb
}

print("\n📈 SECTION 4.2: VEHICLE DIAGNOSTIC MODEL PERFORMANCE MATRIX")
header = f"{'Evaluation Metric':<18} | {'Random Forest':<14} | {'XGBoost':<10}"
print("-" * len(header))
print(header)
print("-" * len(header))
print(f"{'Accuracy':<18} | {accuracy_score(y_test, models['Random Forest']):>12.2%} | {accuracy_score(y_test, models['XGBoost']):>9.2%}")
print(f"{'Precision':<18} | {precision_score(y_test, models['Random Forest']):>12.2%} | {precision_score(y_test, models['XGBoost']):>9.2%}")
print(f"{'Recall Rate':<18} | {recall_score(y_test, models['Random Forest']):>12.2%} | {recall_score(y_test, models['XGBoost']):>9.2%}")
print(f"{'F1-Score':<18} | {f1_score(y_test, models['Random Forest']):>12.4f} | {f1_score(y_test, models['XGBoost']):>9.4f}")
print("-" * len(header))

# Generate and save a scientifically sound confusion matrix plot
cm = confusion_matrix(y_test, y_pred_rf)
from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal (0)', 'Fault (1)'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
ax.set_title('Figure 4.4: Random Forest Edge Evaluation Confusion Matrix')
ax.set_xlabel('Predicted Class')
ax.set_ylabel('True Class')
plt.tight_layout()
fig.savefig('plot_confusion_matrix_final.png', dpi=300)
plt.close(fig)

print("\n🖼️ Visual Graphics successfully exported: 'plot_confusion_matrix_final.png'")
print("="*80)