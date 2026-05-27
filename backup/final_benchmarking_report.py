# final_benchmarking_report.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

print("="*75)
print("🔬 EXECUTING FULL Q1 JOURNAL COMPARATIVE BENCHMARKING PIPELINE")
print("="*75)

# Load the synthetic dataset
df = pd.read_csv('engine_data_synthetic.csv')

features = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
X = df[features].values
y = df['Engine Condition'].values

# Split data using fixed random state for uniform comparison
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 2. XGBoost
xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# 3. Multi-Layer Perceptron (Neural Network)
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

# 4. K-Means (Unsupervised Clustering)
# We map clusters to match the majority true label class
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X_train) # Unsupervised training
clusters = kmeans.predict(X_test)

# Map clusters to actual labels (0 or 1) based on alignment
mapping = {}
for cluster_idx in [0, 1]:
    true_labels_in_cluster = y_test[clusters == cluster_idx]
    if len(true_labels_in_cluster) > 0:
        majority_label = np.bincount(true_labels_in_cluster).argmax()
        mapping[cluster_idx] = majority_label
    else:
        mapping[cluster_idx] = cluster_idx

y_pred_kmeans = np.array([mapping[c] for c in clusters])

# Print side-by-side comparison report for Section 4.2
print("\n📊 SUBSECTION 4.2: VEHICLE DIAGNOSTIC MODEL PERFORMANCE MATRIX")
print("-" * 78)
print(f"{'Evaluation Metric':<18} | {'K-Means':<12} | {'Random Forest':<14} | {'XGBoost':<11} | {'MLP NN':<10}")
print("-" * 78)
print(f"{'Accuracy':<18} | {accuracy_score(y_test, y_pred_kmeans):>10.2%} | {accuracy_score(y_test, y_pred_rf):>12.2%} | {accuracy_score(y_test, y_pred_xgb):>9.2%} | {accuracy_score(y_test, y_pred_mlp):>8.2%}")
print(f"{'Precision':<18} | {precision_score(y_test, y_pred_kmeans):>10.2%} | {precision_score(y_test, y_pred_rf):>12.2%} | {precision_score(y_test, y_pred_xgb):>9.2%} | {precision_score(y_test, y_pred_mlp):>8.2%}")
print(f"{'Recall Rate':<18} | {recall_score(y_test, y_pred_kmeans):>10.2%} | {recall_score(y_test, y_pred_rf):>12.2%} | {recall_score(y_test, y_pred_xgb):>9.2%} | {recall_score(y_test, y_pred_mlp):>8.2%}")
print(f"{'F1-Score':<18} | {f1_score(y_test, y_pred_kmeans):>10.4f} | {f1_score(y_test, y_pred_rf):>12.4f} | {f1_score(y_test, y_pred_xgb):>9.4f} | {f1_score(y_test, y_pred_mlp):>8.4f}")
print("-" * 78)
print("="*75)