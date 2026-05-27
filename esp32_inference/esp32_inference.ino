#include "model_config.h"
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>

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

bool parseCSV(const String &input, float output[NUM_FEATURES]) {
    char buffer[128];
    input.toCharArray(buffer, sizeof(buffer));
    char *token = strtok(buffer, ",");
    int idx = 0;

    while (token != nullptr && idx < NUM_FEATURES) {
        output[idx++] = atof(token);
        token = strtok(nullptr, ",");
    }
    return (idx == NUM_FEATURES);
}

void standardize(const float input[NUM_FEATURES], float output[NUM_FEATURES]) {
    for (int i = 0; i < NUM_FEATURES; i++) {
        output[i] = (input[i] - MEANS[i]) / STDS[i];
    }
}

float euclideanDistance(const float a[NUM_FEATURES], const float b[NUM_FEATURES]) {
    float sum = 0.0f;
    for (int i = 0; i < NUM_FEATURES; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

float minDistanceToCentroids(const float normalized[NUM_FEATURES]) {
    float minDist = INFINITY;
    for (int c = 0; c < NUM_CLUSTERS; c++) {
        float dist = euclideanDistance(normalized, CENTROIDS[c]);
        if (dist < minDist) {
            minDist = dist;
        }
    }
    return minDist;
}

int clampHealth(float value) {
    if (value < 0.0f) return 0;
    if (value > 100.0f) return 100;
    return (int)value;
}

void setup() {
    Serial.begin(115200);
    delay(200);

    Serial.println("========================================");
    Serial.println("ESP32 Edge-AI OBD-II Predictive Maintenance");
    Serial.println("========================================");
    Serial.println("Starting BLE and inference engine...");
    Serial.print("THRESHOLD = ");
    Serial.println(THRESHOLD, 6);

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

    Serial.println("BLE advertising started. Waiting for HIL data...");
}

void loop() {
    if (Serial.available()) {
        String input = Serial.readStringUntil('\n');
        input.trim();
        if (input.length() == 0) {
            return;
        }

        float rawFeatures[NUM_FEATURES];
        if (!parseCSV(input, rawFeatures)) {
            Serial.println("[ERROR] Invalid CSV format received");
            return;
        }

        float normalized[NUM_FEATURES];
        standardize(rawFeatures, normalized);

        unsigned long startMicros = micros();
        float d_min = minDistanceToCentroids(normalized);
        unsigned long latency = micros() - startMicros;

        bool anomaly = (d_min > THRESHOLD);
        int score = clampHealth(100.0f * (1.0f - (d_min / THRESHOLD)));
        if (d_min >= THRESHOLD) {
            score = 0;
        }

        Serial.print("[DATA] H:");
        Serial.print(score);
        Serial.print(" A:");
        Serial.print(anomaly ? 1 : 0);
        Serial.print(" D:");
        Serial.print(d_min, 4);
        Serial.print(" Latency:");
        Serial.print(latency);
        Serial.println(" us");

        if (deviceConnected) {
            char payload[64];
            snprintf(payload, sizeof(payload), "H:%d,A:%d,D:%.4f", score, anomaly ? 1 : 0, d_min);
            pCharacteristic->setValue(payload);
            pCharacteristic->notify();
        }
    }

    delay(10);
}
