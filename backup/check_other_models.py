# check_other_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier

print("="*60)
print("🔬 CROSS-MODEL BENCHMARKING ON SEPARATED PHYSICS DATA")
print("="*60)

# Load the synthetic dataset
df = pd.read_csv('engine_data_synthetic.csv')

features = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
X = df[features].values
y = df['Engine Condition'].values

# Split data (using the exact same random state for absolute fairness)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Initialize and train XGBoost
print("⚡ Training Extreme Gradient Boosting (XGBoost)...")
xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# 2. Initialize and train Multi-Layer Perceptron (Neural Network)
print("🧠 Training Multi-Layer Perceptron (Neural Network)...")
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

# 3. Print side-by-side comparison report
print("\n📊 FINAL COMPARATIVE PERFORMANCE METRICS:")
print("-" * 65)
print(f"{'Evaluation Metric':<20} | {'XGBoost Model':<18} | {'Neural Network (MLP)':<18}")
print("-" * 65)
print(f"{'Accuracy':<20} | {accuracy_score(y_test, y_pred_xgb):>16.2%} | {accuracy_score(y_test, y_pred_mlp):>16.2%}")
print(f"{'Precision':<20} | {precision_score(y_test, y_pred_xgb):>16.2%} | {precision_score(y_test, y_pred_mlp):>16.2%}")
print(f"{'Recall Rate':<20} | {recall_score(y_test, y_pred_xgb):>16.2%} | {recall_score(y_test, y_pred_mlp):>16.2%}")
print(f"{'F1-Score':<20} | {f1_score(y_test, y_pred_xgb):>16.4f} | {f1_score(y_test, y_pred_mlp):>16.4f}")
print("-" * 65)
print("="*60)