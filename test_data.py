# test_data.py - Run this to verify your data
import pandas as pd
import json

# Check your data file
df = pd.read_csv('engine_data_synthetic.csv')
print(f"Data shape: {df.shape}")
print(f"Fault ratio: {(df['Engine Condition']==1).mean():.2%}")
print("\nFirst 5 rows:")
print(df.head())

# Check your model config (if saved as JSON)
try:
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    print("\n✅ Model config loaded")
    print(f"Clusters: {config['num_clusters']}")
    print(f"Threshold: {config['threshold']}")
except:
    print("\n❌ model_config.json not found - check the filename")