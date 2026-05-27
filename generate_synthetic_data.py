# generate_synthetic_data.py
"""Generate synthetic OBD-II normal and anomaly data for training and testing."""

import numpy as np
import pandas as pd

NORMAL_SAMPLES = 15000
ANOMALY_SAMPLES = 5000
OUTPUT_FILE = 'engine_data_synthetic.csv'

FEATURES = [
    'Engine rpm',
    'Lub oil pressure',
    'Fuel pressure',
    'Coolant pressure',
    'lub oil temp',
    'Coolant temp',
    'Engine Condition'
]


def generate_normal_data(n):
    rng = np.random.default_rng(42)
    rpm = rng.uniform(600, 2200, size=n)
    oil_pressure = rng.normal(3.5, 0.6, size=n)
    fuel_pressure = rng.normal(6.5, 1.2, size=n)
    coolant_pressure = rng.normal(2.5, 0.5, size=n)
    oil_temp = rng.normal(80.0, 2.5, size=n)
    coolant_temp = rng.normal(80.0, 2.5, size=n)

    columns = {
        'Engine rpm': np.clip(rpm, 600, 2200),
        'Lub oil pressure': np.clip(oil_pressure, 2.0, 5.0),
        'Fuel pressure': np.clip(fuel_pressure, 3.0, 10.0),
        'Coolant pressure': np.clip(coolant_pressure, 1.0, 4.0),
        'lub oil temp': np.clip(oil_temp, 75.0, 85.0),
        'Coolant temp': np.clip(coolant_temp, 75.0, 85.0),
        'Engine Condition': np.zeros(n, dtype=int)
    }
    return pd.DataFrame(columns)


def generate_anomaly_data(n):
    rng = np.random.default_rng(43)
    rpm = rng.uniform(400, 2500, size=n)
    oil_pressure = rng.normal(2.0, 1.0, size=n)
    fuel_pressure = rng.normal(11.0, 2.5, size=n)
    coolant_pressure = rng.normal(0.8, 0.8, size=n)
    oil_temp = rng.normal(95.0, 8.0, size=n)
    coolant_temp = rng.normal(95.0, 8.0, size=n)

    # Inject occasional irregularities
    spike_indices = rng.choice(n, size=int(n * 0.25), replace=False)
    oil_pressure[spike_indices] *= rng.uniform(0.4, 0.7, size=spike_indices.shape)
    coolant_pressure[spike_indices] += rng.uniform(1.0, 2.5, size=spike_indices.shape)

    columns = {
        'Engine rpm': np.clip(rpm, 400, 2500),
        'Lub oil pressure': np.clip(oil_pressure, 0.0, 6.0),
        'Fuel pressure': np.clip(fuel_pressure, 1.0, 15.0),
        'Coolant pressure': np.clip(coolant_pressure, 0.0, 6.0),
        'lub oil temp': np.clip(oil_temp, 60.0, 110.0),
        'Coolant temp': np.clip(coolant_temp, 60.0, 110.0),
        'Engine Condition': np.ones(n, dtype=int)
    }
    return pd.DataFrame(columns)


def main():
    normal_df = generate_normal_data(NORMAL_SAMPLES)
    anomaly_df = generate_anomaly_data(ANOMALY_SAMPLES)
    df = pd.concat([normal_df, anomaly_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[OK] Generated {len(df)} synthetic samples and saved to {OUTPUT_FILE}")
    print(f"  Normal: {NORMAL_SAMPLES}, Anomaly: {ANOMALY_SAMPLES}")


if __name__ == '__main__':
    main()
