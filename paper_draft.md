# Draft Paper: Edge-AI OBD-II Predictive Maintenance

## Executive Summary

This paper presents an Edge-AI approach for predictive maintenance in automotive OBD-II systems. A supervised Random Forest classifier is trained on a realistic labeled engine dataset with six OBD-II features to identify engine faults. The implementation is supported by a repository that includes benchmark scripts, data generation, and an ESP32-based edge demo branch. The primary contribution is a practical model and deployment workflow for low-cost OBD-II monitoring.

## 1. Introduction

Modern vehicles generate rich OBD-II telemetry that can be used for predictive maintenance. This work demonstrates how supervised machine learning can detect engine faults from a small set of OBD-II signals and how the same pipeline can be adapted into an edge monitoring prototype. The main scientific objective is to show that a Random Forest model trained on realistic engine data can achieve high accuracy for fault detection while keeping the feature set and deployment workflow manageable.

## 2. Related Work

Prior work in vehicle diagnostics includes supervised classification for engine fault detection, unsupervised anomaly detection for maintenance, and TinyML edge deployments for automotive telemetry. This paper positions itself in the intersection of realistic OBD-II fault classification and lightweight edge-system validation, using a supervised classifier as the primary evaluation model.

## 3. Dataset and Features

The dataset used in this study is `engine_data_realistic.csv`, which contains 19,535 labeled engine telemetry records. Each row includes the target field `Engine Condition` and the following six input features:

- `Engine rpm`
- `Lub oil pressure`
- `Fuel pressure`
- `Coolant pressure`
- `lub oil temp`
- `Coolant temp`

The dataset is split into 80% training and 20% test sets using stratified sampling to preserve the class distribution.

## 4. Methodology

### 4.1 Supervised Classification

The primary model is a Random Forest classifier with the following training configuration:

- `n_estimators=100`
- `max_depth=10`
- `random_state=42`
- `n_jobs=-1`

A secondary comparator model is XGBoost, included as a supervised baseline for evaluating relative performance.

### 4.2 Edge Demonstration

The repository also contains an optional edge demo path where an ESP32 firmware prototype analyzes 6 CSV features and emits BLE telemetry. This branch is secondary to the paper’s main supervised classification narrative and serves as an engineering demonstration of end-to-end deployment.

## 5. Experimental Setup

The evaluation uses the realistic labeled dataset and the following protocol:

- Train/test split: 80% train, 20% test
- Stratified sampling on `Engine Condition`
- Performance metrics: Accuracy, Precision, Recall, F1-score
- Main model: Random Forest
- Comparator: XGBoost

The benchmark pipeline is implemented in `run_realistic_benchmarks.py` and generates a confusion matrix image at `plot_confusion_matrix_final.png`.

## 6. Results

### 6.1 Classification Performance

The supervised Random Forest classifier achieves the following performance on the held-out test set:

| Metric | Random Forest | XGBoost |
|---|---|---|
| Accuracy | 93.11% | 92.99% |
| Precision | 93.71% | 94.15% |
| Recall | 95.49% | 94.76% |
| F1-score | 0.9459 | 0.9446 |

### 6.2 Confusion Matrix

The detected test-set confusion matrix is:

| | Predicted Normal | Predicted Fault |
|---|---|---|
| Actual Normal | 1,286 | 158 |
| Actual Fault | 111 | 2,352 |

**Figure 1:** `plot_confusion_matrix_final.png` — Random Forest confusion matrix for the held-out test set.

## 7. Discussion

The Random Forest model demonstrates strong diagnostic performance on the realistic dataset, with very low false negative count (111) and high recall (95.49%). This indicates reliable fault capture, which is critical for predictive maintenance systems.

The XGBoost comparator confirms that the feature set and data preparation are robust, with nearly equivalent performance. The results support the selection of supervised classification for high-confidence fault detection when labeled data is available.

## 8. Conclusion

This work shows that supervised Random Forest classification can provide accurate engine fault detection from a compact set of OBD-II features. The repository supports the research with code, dataset handling, and a practical benchmark script. An optional ESP32 edge demo branch is included for future work on deployment and BLE telemetry.

## 9. Future Work

- Extend the dataset with additional real-world OBD-II fault types.
- Evaluate model generalization across different engine models and vehicles.
- Integrate the supervised classifier into an ESP32 or mobile-assisted edge deployment pipeline.
- Compare end-to-end latency and power consumption for embedded inference.

## Appendix: Implementation Notes

- The paper-relevant benchmark code is in `run_realistic_benchmarks.py`.
- The optional edge demo branch uses `train_and_export.py`, `esp32_inference.ino`, and `ble_monitor.py`.
- The main image file for the paper is `plot_confusion_matrix_final.png`.
