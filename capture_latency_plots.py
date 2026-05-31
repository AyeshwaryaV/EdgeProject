import serial
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === CONFIGURATION ===
PORT = 'COM3'      # Change to your COM port if different
BAUD = 115200
DURATION = 30      # seconds to capture
CSV_FILE = 'engine_data_realistic.csv'
# ====================

# Open serial port
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

# Load data
df = pd.read_csv(CSV_FILE)
print(f'Loaded {len(df)} rows from {CSV_FILE}')

# Column names from your CSV
rpm_col = 'Engine rpm'
oil_pressure_col = 'Lub oil pressure'
fuel_pressure_col = 'Fuel pressure'
coolant_pressure_col = 'Coolant pressure'
oil_temp_col = 'lub oil temp'
coolant_temp_col = 'Coolant temp'

latencies = []
print(f'Sending data and capturing latency for {DURATION} seconds...')
start_time = time.time()
row_idx = 0
total_rows = len(df)

while time.time() - start_time < DURATION:
    if row_idx >= total_rows:
        row_idx = 0
    row = df.iloc[row_idx]
    
    # Create CSV line with correct column names
    line = f"{row[rpm_col]},{row[oil_pressure_col]},{row[fuel_pressure_col]},{row[coolant_pressure_col]},{row[oil_temp_col]},{row[coolant_temp_col]}\n"
    ser.write(line.encode())
    
    timeout_end = time.time() + 0.1
    while time.time() < timeout_end:
        if ser.in_waiting:
            resp = ser.readline().decode('utf-8', errors='ignore')
            if 'Latency:' in resp:
                try:
                    parts = resp.split('Latency:')
                    lat_us = int(parts[1].split('us')[0].strip())
                    latencies.append(lat_us)
                    print(f'Latency: {lat_us} us')
                except:
                    pass
    row_idx += 1
    time.sleep(0.45)

ser.close()

if latencies:
    print(f'\n✅ Captured {len(latencies)} samples')
    print(f'Mean: {np.mean(latencies):.1f} us')
    print(f'Min: {np.min(latencies)} us')
    print(f'Max: {np.max(latencies)} us')
    print(f'95th Percentile: {np.percentile(latencies, 95):.1f} us')

    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    axes[0].hist(latencies, bins=20, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Latency (microseconds)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('ESP32 Inference Latency Distribution')
    axes[0].axvline(np.mean(latencies), color='red', linestyle='--', label=f'Mean: {np.mean(latencies):.1f} µs')
    axes[0].legend()

    sorted_lat = np.sort(latencies)
    cdf = np.arange(1, len(sorted_lat)+1) / len(sorted_lat)
    axes[1].plot(sorted_lat, cdf, 'b-', linewidth=2)
    axes[1].set_xlabel('Latency (microseconds)')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('CDF of Inference Latency')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0.95, color='red', linestyle='--', label='95% Threshold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('latency_plots.png', dpi=300)
    print('\n✅ Saved: latency_plots.png (Figures 4.4 & 4.5)')
else:
    print('No latency data captured.') 