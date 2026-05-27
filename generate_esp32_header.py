# generate_esp32_header.py
import json
import os

print("="*50)
print("Step 2: Generating ESP32 C++ Header")
print("="*50)

# Load config
with open('model_config.json', 'r') as f:
    config = json.load(f)

num_features = config['num_features']
num_clusters = config['num_clusters']

# Create folder
os.makedirs('esp32_inference', exist_ok=True)

# Generate header file
header = f"""// model_config.h - Auto-generated from trained model
#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#include <Arduino.h>
#include <math.h>

// Model configuration
const int NUM_FEATURES = {num_features};
const int NUM_CLUSTERS = {num_clusters};
const float ANOMALY_THRESHOLD = {config['threshold']:.6f}f;

// Standardization parameters (mean and std)
const float SCALER_MEAN[NUM_FEATURES] = {{
"""
for i, val in enumerate(config['scaler_mean']):
    header += f"    {val:.6f}f"
    if i < num_features-1:
        header += ","
    header += "\n"
header += "};\n\nconst float SCALER_STD[NUM_FEATURES] = {\n"
for i, val in enumerate(config['scaler_std']):
    header += f"    {val:.6f}f"
    if i < num_features-1:
        header += ","
    header += "\n"
header += "};\n\n// Cluster centroids (normalized space)\nconst float CENTROIDS[NUM_CLUSTERS][NUM_FEATURES] = {\n"

for i, centroid in enumerate(config['centroids']):
    header += "    { "
    for j, val in enumerate(centroid):
        header += f"{val:.6f}f"
        if j < num_features-1:
            header += ", "
    header += " }"
    if i < num_clusters-1:
        header += ","
    header += "\n"
header += "};\n\n#endif // MODEL_CONFIG_H\n"

# Save header
with open('esp32_inference/model_config.h', 'w') as f:
    f.write(header)

print("[OK] Created esp32_inference/model_config.h")
print("\nNext: Copy this header to your ESP32 project")

