# export_for_esp32.py - FIXED (no emojis)
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import os

def export_from_trained_model():
    """Export model parameters from trained data"""
    
    # Select the best available dataset
    candidate_files = ['engine_data.csv', 'engine_data_synthetic.csv']
    for fname in candidate_files:
        if os.path.exists(fname):
            data_file = fname
            break
    else:
        raise FileNotFoundError('No engine_data.csv or engine_data_synthetic.csv found in project root.')

    print(f"[INFO] Using dataset: {data_file}")
    df = pd.read_csv(data_file)
    df.columns = [c.strip() for c in df.columns]
    feature_cols = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 
                    'Coolant pressure', 'lub oil temp', 'Coolant temp']

    if 'Engine Condition' not in df.columns:
        raise ValueError('Dataset must contain an Engine Condition label column for normal-only training.')

    normal_df = df[df['Engine Condition'] == 0].copy()
    if len(normal_df) == 0:
        raise ValueError('No normal samples found in the dataset to train the edge model.')

    X = normal_df[feature_cols].values
    
    # Train scaler and KMeans
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Calculate threshold (95th percentile of distances)
    distances = kmeans.transform(X_scaled).min(axis=1)
    threshold = np.percentile(distances, 95)
    
    # Export to JSON
    config = {
        'feature_columns': feature_cols,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
        'centroids': kmeans.cluster_centers_.tolist(),
        'threshold': float(threshold),
        'num_clusters': 3,
        'num_features': 6
    }
    
    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("[OK] Saved model_config.json")
    
    # Generate ESP32 C++ header
    generate_esp32_header(config)

def generate_esp32_header(config):
    """Generate C++ header file for ESP32"""
    
    num_features = config['num_features']
    num_clusters = config['num_clusters']
    
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
    
    os.makedirs('esp32_inference', exist_ok=True)
    
    with open('esp32_inference/model_config.h', 'w') as f:
        f.write(header)
    
    print("[OK] Saved esp32_inference/model_config.h")

def update_esp32_sketch():
    """Update the main ESP32 sketch"""
    
    sketch_content = '''// esp32_inference.ino - BLE-Enabled Edge-AI OBD-II
#include "model_config.h"
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>

// BLE Configuration
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

BLECharacteristic *pCharacteristic;
bool deviceConnected = false;

// BLE Server Callbacks
class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) {
        deviceConnected = true;
        Serial.println("[BLE] Client Connected");
    }
    void onDisconnect(BLEServer* pServer) {
        deviceConnected = false;
        Serial.println("[BLE] Client Disconnected");
        pServer->startAdvertising();
    }
};

// Parse CSV input: "val1,val2,val3,val4,val5,val6"
bool parseCSV(String input, float output[NUM_FEATURES]) {
    int idx = 0;
    char* ptr = strtok((char*)input.c_str(), ",");
    while (ptr != NULL && idx < NUM_FEATURES) {
        output[idx++] = atof(ptr);
        ptr = strtok(NULL, ",");
    }
    return (idx == NUM_FEATURES);
}

// Standardize features using training statistics
void standardize(float input[NUM_FEATURES], float output[NUM_FEATURES]) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        output[i] = (input[i] - SCALER_MEAN[i]) / SCALER_STD[i];
    }
}

// Euclidean distance between two points
float euclideanDistance(float* a, float* b) {
    float sum = 0.0;
    for (int i = 0; i < NUM_FEATURES; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Find minimum distance to any cluster centroid
float minDistanceToCentroids(float normalized[NUM_FEATURES]) {
    float minDist = INFINITY;
    for (int c = 0; c < NUM_CLUSTERS; c++) {
        float dist = euclideanDistance(normalized, (float*)CENTROIDS[c]);
        if (dist < minDist) minDist = dist;
    }
    return minDist;
}

void setup() {
    Serial.begin(115200);
    delay(200);
    
    Serial.println("\\n========================================");
    Serial.println("ESP32 Edge-AI OBD-II Anomaly Detector");
    Serial.println("========================================");
    Serial.print("Model: ");
    Serial.print(NUM_CLUSTERS);
    Serial.println("-Cluster KNN");
    Serial.print("Threshold: ");
    Serial.println(ANOMALY_THRESHOLD);
    
    // Initialize BLE
    BLEDevice::init("ESP32-OBD-Monitor");
    BLEServer *pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());
    
    BLEService *pService = pServer->createService(SERVICE_UUID);
    pCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID,
        BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY
    );
    pCharacteristic->addDescriptor(new BLE2902());
    
    pService->start();
    pServer->getAdvertising()->start();
    
    Serial.println("BLE Active - Ready for HIL Data\\n");
}

void loop() {
    if (Serial.available()) {
        String input = Serial.readStringUntil('\\n');
        input.trim();
        if (input.length() == 0) return;
        
        float rawFeatures[NUM_FEATURES];
        if (!parseCSV(input, rawFeatures)) {
            Serial.println("[ERROR] Parse Error");
            return;
        }
        
        unsigned long startTime = micros();
        
        float normalized[NUM_FEATURES];
        standardize(rawFeatures, normalized);
        
        float distance = minDistanceToCentroids(normalized);
        bool isAnomaly = (distance > ANOMALY_THRESHOLD);
        
        // Health score: 100% = normal, 0% = severe anomaly
        int healthScore = (int)(100.0f * (1.0f - min(1.0f, distance / ANOMALY_THRESHOLD)));
        
        unsigned long latency = micros() - startTime;
        
        // Serial output for HIL recording
        Serial.print("[DATA] Health:");
        Serial.print(healthScore);
        Serial.print("% | ");
        Serial.print(isAnomaly ? "ANOMALY" : "NORMAL");
        Serial.print(" | Dist:");
        Serial.print(distance, 3);
        Serial.print(" | Latency:");
        Serial.print(latency);
        Serial.println(" us");
        
        // BLE notification
        if (deviceConnected) {
            char bleBuffer[50];
            snprintf(bleBuffer, sizeof(bleBuffer), "H:%d,A:%d,D:%.2f", 
                     healthScore, isAnomaly ? 1 : 0, distance);
            pCharacteristic->setValue(bleBuffer);
            pCharacteristic->notify();
        }
    }
    delay(10);
}
'''
    
    os.makedirs('esp32_inference', exist_ok=True)
    
    with open('esp32_inference/esp32_inference.ino', 'w', encoding='utf-8') as f:
        f.write(sketch_content)
    
    print("[OK] Updated esp32_inference/esp32_inference.ino")

def create_hil_player():
    """Create the HIL player script if it doesn't exist"""
    
    hil_content = '''"""
hil_player.py - Streams engine data to ESP32 over USB Serial
Run this AFTER uploading firmware to ESP32
"""

import serial
import time
import pandas as pd
import sys

# ============================================
# CONFIGURATION - CHANGE THIS
# ============================================
SERIAL_PORT = 'COM3'  # Change to your ESP32 port
# Windows: 'COM3', 'COM4', etc.
# Linux/Mac: '/dev/ttyUSB0' or '/dev/ttyACM0'
# ============================================

def main():
    print("="*60)
    print("HIL Telemetry Player")
    print("="*60)
    
    # Load dataset
    try:
        df = pd.read_csv('engine_data_synthetic.csv')
    except FileNotFoundError:
        try:
            df = pd.read_csv('engine_data.csv')
        except FileNotFoundError:
            print("[ERROR] No data file found!")
            return
    
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # Features to stream
    feature_cols = ['Engine rpm', 'Lub oil pressure', 'Fuel pressure', 
                    'Coolant pressure', 'lub oil temp', 'Coolant temp']
    
    # Check columns
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns: {missing}")
        return
    
    data = df[feature_cols].copy()
    data = data.dropna()
    
    print(f"[OK] Loaded {len(data)} rows")
    print(f"[INFO] Features: {feature_cols}")
    print(f"[INFO] Serial port: {SERIAL_PORT}")
    
    # Open serial connection
    try:
        ser = serial.Serial(SERIAL_PORT, 115200, timeout=1)
        time.sleep(2)
        print(f"[OK] Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"[ERROR] Cannot open {SERIAL_PORT}: {e}")
        return
    
    print("\\n[INFO] Streaming data to ESP32 at 2Hz...")
    print("Press Ctrl+C to stop\\n")
    print("="*60)
    
    try:
        for idx, row in data.iterrows():
            data_string = f"{row['Engine rpm']},{row['Lub oil pressure']},{row['Fuel pressure']},{row['Coolant pressure']},{row['lub oil temp']},{row['Coolant temp']}\\n"
            ser.write(data_string.encode())
            
            # Read ESP32 response
            if ser.in_waiting:
                response = ser.readline().decode().strip()
                print(f"[{idx+1:4d}] Sent | {response}")
            else:
                print(f"[{idx+1:4d}] Sent | (no response)")
            
            time.sleep(0.5)  # 2Hz = 500ms
            
    except KeyboardInterrupt:
        print("\\n\\n[STOPPED] By user")
    finally:
        ser.close()
        print("[INFO] Serial closed")

if __name__ == "__main__":
    main()
'''
    
    if not os.path.exists('hil_player.py'):
        with open('hil_player.py', 'w', encoding='utf-8') as f:
            f.write(hil_content)
        print("[OK] Created hil_player.py")
    else:
        print("[INFO] hil_player.py already exists")

if __name__ == "__main__":
    print("="*60)
    print("ESP32 Model Export Tool")
    print("="*60)
    
    export_from_trained_model()
    update_esp32_sketch()
    create_hil_player()
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Open esp32_inference/esp32_inference.ino in Arduino IDE")
    print("2. Select ESP32 Dev Module as board")
    print("3. Install required libraries (BLEDevice is built-in)")
    print("4. Compile and upload to ESP32")
    print("5. Run: python hil_player.py")
    print("="*60)