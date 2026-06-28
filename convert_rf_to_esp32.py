# convert_rf_to_esp32.py
import pandas as pd
import emlearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("=" * 50)
print("Converting Random Forest to ESP32 C++ Header")
print("=" * 50)

# Load dataset
df = pd.read_csv('engine_data_realistic.csv')
feature_cols = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 
                'Coolant pressure', 'lub oil temp', 'Coolant temp']
X = df[feature_cols]
y = df['Engine Condition']

# Train Random Forest (small for ESP32)
rf = RandomForestClassifier(
    n_estimators=20,      # Small size for ESP32 flash
    max_depth=8,          # Limits tree complexity
    random_state=42
)
rf.fit(X, y)

# Evaluate
accuracy = rf.score(X, y)
print(f"✅ Random Forest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Convert to C code
cmodel = emlearn.convert(rf, method='inline')
cmodel.save(file='random_forest_model.h')

print("✅ Saved: random_forest_model.h")
print("📁 This header can now be included in your ESP32 sketch")