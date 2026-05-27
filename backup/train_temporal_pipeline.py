# train_temporal_pipeline.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

print("="*60)
print("⏳ RUNNING TEMPORAL WINDOW SLIDING FEATURE PIPELINE")
print("="*60)

# 1. Load data
df = pd.read_csv('engine_data.csv')
df.columns = [c.strip() for c in df.columns]

feature_cols = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']

# 2. Extract Window-Based Behavioral Trends
print("⚙️ Processing rolling time horizons (Window Size = 10)...")
window_size = 10
stride = 1  # Continuous stride to preserve maximum vector density

X_temporal = []
y_labels = []

# Convert to numpy for ultra-fast row slicing
raw_matrix = df[feature_cols].values
raw_labels = df['Engine Condition'].values

for i in range(0, len(df) - window_size, stride):
    window = raw_matrix[i : i + window_size]
    features = []
    
    for col_idx in range(len(feature_cols)):
        series = window[:, col_idx]
        
        # Extract DeepSeek's recommended temporal vectors
        current_val = series[-1]
        slope = series[-1] - series[0]  # Trend direction over window boundary
        volatility = np.std(series)      # Instability metric
        acceleration = series[-1] - series[-2] # Instantaneous slope change
        
        features.extend([current_val, slope, volatility, acceleration])
        
    X_temporal.append(features)
    # Target label reflects current state at end of window sequence
    y_labels.append(raw_labels[i + window_size - 1])

X = np.array(X_temporal)
y = np.array(y_labels)

print(f" ├─ Input Features Generated per Window: {X.shape[1]}")
print(f" └─ Total Sequence Windows Extracted:    {len(X)}")

# 3. Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Highly Optimized Edge Classifier
print("\n🏋️ Training sliding-window classification network...")
edge_model = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=22,
    random_state=42,
    n_jobs=-1
)
edge_model.fit(X_train_scaled, y_train)

# 5. Evaluate System Performance
y_pred = edge_model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n🎯 NEW TEMPORAL SLIDING WINDOW RESULTS:")
print(f" ├─ Target Accuracy:  {acc:.2%}")
print(f" ├─ System Precision: {prec:.2%}")
print(f" ├─ System Recall:    {rec:.2%}")
print(f" └─ Journal F1-Score: {f1:.4f}")

print("\n📋 TIMELINE MATRICES DIAGNOSTIC PROFILE:")
print(f" True Normals Caught (TN): {cm[0,0]} | False Alarms (FP): {cm[0,1]}")
print(f" Missed Faults (FN):         {cm[1,0]} | Actual Faults Caught (TP): {cm[1,1]}")

# Save the updated confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', cbar=True,
            xticklabels=['Predicted Normal (0)', 'Predicted Anomaly (1)'],
            yticklabels=['Actual Normal (0)', 'Actual Anomaly (1)'])
plt.title('Figure 4.4: Sliding-Window Temporal Confusion Matrix')
plt.ylabel('True Class Label')
plt.xlabel('Edge System Prediction')
plt.tight_layout()
plt.savefig('plot_confusion_matrix_temporal.png', dpi=300)

# Save the updated ROC Curve Plot
plt.figure(figsize=(6, 5))
y_probs = edge_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='crimson', lw=2, label=f'ROC Curve (Area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Figure 4.5: Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('plot_roc_curve_temporal.png', dpi=300)

print("\n🖼️ Verification charts generated successfully:")
print(" ├─ plot_confusion_matrix_temporal.png")
print(" └─ plot_roc_curve_temporal.png")
print("="*60)