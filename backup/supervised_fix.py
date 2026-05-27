# supervised_fix.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('engine_data.csv')
df.columns = [c.strip() for c in df.columns]

# Features (all except Engine Condition)
feature_cols = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 
                'Coolant pressure', 'lub oil temp', 'Coolant temp']

X = df[feature_cols].values
y = df['Engine Condition'].values

print("="*60)
print("DATASET INFO")
print("="*60)
print(f"Total samples: {len(y)}")
print(f"Normal (0): {(y==0).sum()}")
print(f"Anomaly (1): {(y==1).sum()}")
print(f"Anomaly ratio: {(y==1).sum()/len(y)*100:.1f}%")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
print("\n" + "="*60)
print("TRAINING RANDOM FOREST")
print("="*60)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)

# Predict
y_pred = rf.predict(X_test_scaled)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\n✅ Accuracy: {acc:.2%}")
print(f"✅ Precision: {prec:.2%}")
print(f"✅ Recall: {rec:.2%}")
print(f"✅ F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
print("              Predicted")
print("              Normal  Anomaly")
print(f"Actual Normal  {cm[0,0]:5d}   {cm[0,1]:5d}")
print(f"       Anomaly {cm[1,0]:5d}   {cm[1,1]:5d}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))

# Cross-validation
cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)
print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

# Feature importance
importances = rf.feature_importances_
print("\nFeature Importances:")
for name, imp in zip(feature_cols, importances):
    print(f"  {name}: {imp:.2%}")

# Confusion Matrix Plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png', dpi=150)
print("\n✅ Confusion matrix saved as 'rf_confusion_matrix.png'")

# Publication readiness check
print("\n" + "="*60)
print("PUBLICATION READINESS CHECK")
print("="*60)
if acc >= 0.85:
    print("✅ Accuracy >=85% - Good for Q1 Journal")
elif acc >= 0.75:
    print("⚠️ Accuracy between 75-85% - Acceptable for M.Tech, borderline for Q1")
else:
    print("❌ Accuracy <75% - Need improvement")

if f1 >= 0.80:
    print("✅ F1 Score >=0.80 - Good for Q1 Journal")
elif f1 >= 0.70:
    print("⚠️ F1 Score between 0.70-0.80 - Acceptable for M.Tech")
else:
    print("❌ F1 Score <0.70 - Need improvement")

if cm[1,1] > cm[1,0]:
    print("✅ Anomaly detection working well (more TP than FN)")
else:
    print(f"⚠️ Anomaly detection needs improvement (TP={cm[1,1]}, FN={cm[1,0]})")