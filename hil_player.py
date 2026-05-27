# hil_player.py
"""HIL player for streaming OBD-II telemetry to ESP32 via USB serial."""

import argparse
import csv
import os
import time
from datetime import datetime

import pandas as pd
import serial

DATA_FILES = ['engine_data.csv', 'engine_data_synthetic.csv']
FEATURE_COLS = [
    'Engine rpm',
    'Lub oil pressure',
    'Fuel pressure',
    'Coolant pressure',
    'lub oil temp',
    'Coolant temp'
]


def parse_args():
    parser = argparse.ArgumentParser(description='Stream OBD-II telemetry rows over USB serial to ESP32.')
    parser.add_argument('--data', help='Path to dataset CSV file')
    parser.add_argument('--port', default=os.environ.get('SERIAL_PORT', 'COM3'), help='ESP32 serial port (default: COM3 or SERIAL_PORT env)')
    parser.add_argument('--baud', type=int, default=115200, help='Serial baud rate')
    parser.add_argument('--rate', type=float, default=0.5, help='Interval between rows, seconds')
    parser.add_argument('--no-log', action='store_true', help='Disable local CSV logging')
    return parser.parse_args()


def locate_dataset(data_file=None):
    if data_file:
        if os.path.exists(data_file):
            return data_file
        raise FileNotFoundError(f"Dataset file '{data_file}' not found.")

    for fname in DATA_FILES:
        if os.path.exists(fname):
            return fname
    raise FileNotFoundError(f"None of {DATA_FILES} exists in the project root.")


def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing_columns = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required feature columns: {missing_columns}")
    return df[FEATURE_COLS].dropna().reset_index(drop=True)


def stream_to_esp32(df, serial_port, baud_rate, interval_s, log_to_csv):
    print(f"[INFO] Opening serial port {serial_port} at {baud_rate} baud")
    ser = serial.Serial(serial_port, baud_rate, timeout=1)
    time.sleep(2)
    ser.reset_input_buffer()
    print('[OK] Serial port opened')

    log_file = None
    csv_writer = None
    if log_to_csv:
        log_filename = f'health_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        log_file = open(log_filename, 'w', newline='')
        csv_writer = csv.writer(log_file)
        csv_writer.writerow(['timestamp', 'row', 'rpm', 'payload', 'esp32_response'])
        print(f"[OK] Logging enabled -> {log_filename}")

    print('[INFO] Streaming rows to ESP32. Press Ctrl+C to stop.')
    print('=' * 72)

    try:
        for idx, row in df.iterrows():
            payload = ','.join(str(row[c]) for c in FEATURE_COLS) + '\n'
            ser.write(payload.encode('utf-8'))
            print(f"[{idx+1:05d}] SENT: {payload.strip()}")

            response = ''
            time.sleep(0.05)
            while ser.in_waiting:
                line = ser.readline().decode(errors='ignore').strip()
                if line:
                    response = line
                    print(f"        RECV: {line}")

            if log_to_csv and csv_writer is not None:
                csv_writer.writerow([datetime.now().isoformat(), idx + 1, row['Engine rpm'], payload.strip(), response])

            time.sleep(interval_s)

    except KeyboardInterrupt:
        print('\n[STOPPED] User interrupted the stream')
    finally:
        ser.close()
        if log_file:
            log_file.close()
            print('[OK] Saved log file')
        print('[INFO] Serial connection closed')


def main():
    args = parse_args()

    print('=' * 72)
    print('EDGE-AI HIL TELEMETRY PLAYER')
    print('=' * 72)

    data_file = locate_dataset(args.data)
    df = load_data(data_file)
    print(f"[OK] Loaded {len(df)} rows from {data_file}")

    stream_to_esp32(df, args.port, args.baud, args.rate, not args.no_log)


if __name__ == '__main__':
    main()
