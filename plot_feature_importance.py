import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('engine_data_realistic.csv')
feature_cols = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
X = df[feature_cols]
y = df['Engine Condition']

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X, y)
importance = rf.feature_importances_
features = ['RPM', 'Oil Pressure', 'Fuel Pressure', 'Coolant Pressure', 'Oil Temp', 'Coolant Temp']

sorted_idx = sorted(range(len(importance)), key=lambda i: importance[i])
plt.figure(figsize=(10,6))
plt.barh([features[i] for i in sorted_idx], [importance[i] for i in sorted_idx], color='steelblue')
plt.xlabel('Feature Importance Score')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print('✅ Figure 4.9 Saved: feature_importance.png')