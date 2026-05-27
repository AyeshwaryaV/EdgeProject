# Edge-AI OBD-II Predictive Maintenance

## Hardware
- ESP32-WROOM-32D
- Laptop/PC for training and HIL streaming
- Mobile phone for BLE telemetry visualization

## Project Files
- `run_realistic_benchmarks.py` - trains supervised Random Forest on realistic labeled engine data and exports paper-ready evaluation metrics
- `train_and_export.py` - optional K-Means/ESP32 export path for the repo’s edge demonstration branch
- `hil_player.py` - streams OBD-II rows over USB serial at 2 Hz to ESP32
- `esp32_inference.ino` - ESP32 firmware for BLE-based anomaly detection and health scoring
- `ble_monitor.py` - reference BLE receiver for health telemetry notifications
- `generate_synthetic_data.py` - optional synthetic dataset generation for normal and anomaly data

## Requirements
- Python 3.9+ with `pandas`, `numpy`, `scikit-learn`, `pyserial`, `bleak`
- Install Python dependencies with:
  ```bash
  pip install pandas numpy scikit-learn pyserial bleak
  ```
- ESP32 board support installed in Arduino IDE
- USB serial cable for ESP32 and laptop connection

## Workflow
1. Prepare the dataset in `engine_data_realistic.csv` or generate synthetic data with `python generate_synthetic_data.py`.
2. Run `python run_realistic_benchmarks.py` to compute IEEE paper metrics from the labeled realistic dataset.
3. Confirm the results printed by the script and the generated `plot_confusion_matrix_final.png` figure.
4. Optionally, use `python train_and_export.py` for the repository’s K-Means/ESP32 edge demonstration branch, but this is not the main paper model.
5. Open `esp32_inference.ino` in Arduino IDE and flash to the ESP32 if you want to test the edge demo.
6. Connect the ESP32 to the laptop and run `python hil_player.py --port COM3`.
7. Use `python ble_monitor.py` or a BLE mobile viewer app to connect to `OBD_Health_Monitor` and receive notifications.

## Notes
- The core IEEE paper model is a supervised Random Forest classifier trained using labeled realistic engine data and 6 OBD-II features.
- `train_and_export.py` remains an optional artifact for the repo’s ESP32/K-Means proof-of-concept branch, but the paper should focus on the Random Forest classification results.
- The ESP32 firmware reads 6 CSV features, standardizes them, computes Euclidean distance to the nearest cluster centroid, then maps the result to a health score and anomaly flag for the edge demo.
- The BLE service UUID is `4fafc201-1fb5-459e-8fcc-c5c9c331914b`.
- The BLE characteristic UUID is `beb5483e-36e1-4688-b7f5-ea07361b26a8`.
- BLE payload format: `H:{score},A:{flag},D:{distance}`.
- Use `ble_monitor.py` as a desktop reference BLE receiver for telemetry validation.

## Mobile App Blueprint
- Connect via BLE to `OBD_Health_Monitor`.
- Subscribe to the characteristic above.
- Display health score with colors:
  - GREEN: 80-100%
  - YELLOW: 50-80%
  - RED: <50%
- Alert when health drops below 50%.
- Keep the last 50 values in a trend graph.
- Allow CSV export of the history.

## Validation
- `python -m py_compile train_and_export.py hil_player.py ble_monitor.py` should run without syntax errors.
- `python generate_synthetic_data.py` produces `engine_data_synthetic.csv`.
- `python train_and_export.py` produces `model_config.h` and `model_config.json`.
- `python ble_monitor.py` should connect to the running ESP32 BLE service and print telemetry.

## Important
This project is designed to work locally without cloud connectivity. The ESP32 performs edge inference and sends BLE health telemetry to the phone.
