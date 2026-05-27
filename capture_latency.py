# capture_latency.py
import serial
import re
import numpy as np
import matplotlib.pyplot as plt

ser = serial.Serial('COM3', 115200, timeout=1)
latencies = []

print("Capturing latency data for 30 seconds...")
import time
start = time.time()
while time.time() - start < 30:
    if ser.in_waiting:
        line = ser.readline().decode().strip()
        latency_match = re.search(r'Latency:\s*(\d+)\s*us', line)
        if latency_match:
            latencies.append(int(latency_match.group(1)))

ser.close()

print(f"\nCaptured {len(latencies)} latency samples")
print(f"Mean: {np.mean(latencies):.1f} us")
print(f"Median: {np.median(latencies):.1f} us")
print(f"Min: {np.min(latencies)} us")
print(f"Max: {np.max(latencies)} us")
print(f"Std Dev: {np.std(latencies):.1f} us")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
axes[0].hist(latencies, bins=20, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Inference Latency (microseconds)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('ESP32 Inference Latency Distribution')
axes[0].axvline(np.mean(latencies), color='red', linestyle='--', label=f'Mean: {np.mean(latencies):.1f}us')
axes[0].legend()

# CDF
sorted_lat = np.sort(latencies)
cdf = np.arange(1, len(sorted_lat)+1) / len(sorted_lat)
axes[1].plot(sorted_lat, cdf, marker='.', linestyle='none', markersize=2)
axes[1].set_xlabel('Inference Latency (microseconds)')
axes[1].set_ylabel('Cumulative Probability')
axes[1].set_title('Cumulative Distribution Function (CDF)')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('latency_analysis.png', dpi=300)
print("✅ Saved: latency_analysis.png")