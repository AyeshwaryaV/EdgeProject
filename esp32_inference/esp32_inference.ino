#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>
#include "random_forest_model.h"

#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

BLECharacteristic *pCharacteristic;
bool deviceConnected = false;

class MyServerCallbacks: public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) override {
        deviceConnected = true;
        Serial.println("[BLE] Client Connected");
    }
    void onDisconnect(BLEServer* pServer) override {
        deviceConnected = false;
        Serial.println("[BLE] Client Disconnected");
        pServer->startAdvertising();
    }
};

void setup() {
    Serial.begin(115200);
    delay(1000);
    
    Serial.println("========================================");
    Serial.println("ESP32 Random Forest Edge-AI");
    Serial.println("OBD-II Predictive Maintenance");
    Serial.println("========================================");
    
    BLEDevice::init("OBD_Health_Monitor");
    BLEServer *pServer = BLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());
    
    BLEService *pService = pServer->createService(SERVICE_UUID);
    pCharacteristic = pService->createCharacteristic(
        CHARACTERISTIC_UUID,
        BLECharacteristic::PROPERTY_NOTIFY
    );
    pCharacteristic->addDescriptor(new BLE2902());
    
    pService->start();
    pServer->getAdvertising()->start();
    
    Serial.println("BLE active: OBD_Health_Monitor");
    Serial.println("Waiting for OBD-II data...");
}

void loop() {
    if (Serial.available()) {
        String input = Serial.readStringUntil('\n');
        input.trim();
        if (input.length() == 0) return;
        
        // Parse CSV (6 features)
        int16_t features[6];
        char buffer[256];
        input.toCharArray(buffer, sizeof(buffer));
        char *token = strtok(buffer, ",");
        int idx = 0;
        while (token != nullptr && idx < 6) {
            features[idx++] = (int16_t)atof(token);
            token = strtok(nullptr, ",");
        }
        
        if (idx != 6) {
            Serial.println("[ERROR] Invalid CSV");
            return;
        }
        
        // Measure latency
        unsigned long startMicros = micros();
        
        // Random Forest prediction
        int prediction = random_forest_model_predict(features, 6);
        
        unsigned long latency = micros() - startMicros;
        
        int healthScore = (prediction == 0) ? 100 : 0;
        
        Serial.print("[DATA] H:");
        Serial.print(healthScore);
        Serial.print(" A:");
        Serial.print(prediction);
        Serial.print(" Latency:");
        Serial.print(latency);
        Serial.println(" us");
        
        if (deviceConnected) {
            char payload[64];
            snprintf(payload, sizeof(payload), "H:%d,A:%d", healthScore, prediction);
            pCharacteristic->setValue(payload);
            pCharacteristic->notify();
        }
    }
    delay(5);
}

