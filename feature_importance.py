import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('engine_data_realistic.csv')
feature_cols = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 'Coolant pressure', 'lub oil temp', 'Coolant temp']
X = df[feature_cols]
y = df['Engine Condition']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importance = rf.feature_importances_
features = ['RPM', 'Oil Press', 'Fuel Press', 'Cool Press', 'Oil Temp', 'Cool Temp']

plt.figure(figsize=(8,5))
plt.barh(features, importance, color='steelblue')
plt.xlabel('Feature Importance Score')
plt.title('Random Forest Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print('Saved: feature_importance.png')