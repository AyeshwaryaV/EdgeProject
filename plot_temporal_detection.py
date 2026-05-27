# plot_temporal_detection.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your HIL player output (modify to save to CSV first)
# For now, create a sample based on your observed pattern
# Your data shows health scores varying: 0, 64, 0, 38, 51, 25, 0, 47, 0, 47, 0, 42, 0, 52, 0, 30, 0, 46, 20, 0, 45, 0, 51, 68, 0, 58, 66, 0

health_sequence = [0, 64, 0, 38, 51, 25, 0, 47, 0, 47, 0, 42, 0, 52, 0, 30, 0, 46, 20, 0, 45, 0, 51, 68, 0, 58, 66, 0]
rpm_sequence = [700, 1221, 716, 845, 824, 1230, 538, 767, 838, 495, 1341, 699, 436, 796, 591, 1053, 646, 659, 751, 591, 1046, 829, 961, 696, 852, 901, 740, 654]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Health score over time
axes[0].plot(health_sequence, 'b-', linewidth=2, marker='o', markersize=4)
axes[0].axhline(y=50, color='orange', linestyle='--', label='Warning Threshold (50%)')
axes[0].axhline(y=30, color='red', linestyle='--', label='Critical Threshold (30%)')
axes[0].fill_between(range(len(health_sequence)), 0, 100, where=[h<50 for h in health_sequence], alpha=0.3, color='orange')
axes[0].fill_between(range(len(health_sequence)), 0, 100, where=[h<30 for h in health_sequence], alpha=0.3, color='red')
axes[0].set_ylabel('Vehicle Health Score (%)')
axes[0].set_title('Real-time Health Score Degradation Detection')
axes[0].legend(loc='upper right')
axes[0].set_ylim(0, 100)
axes[0].grid(True, alpha=0.3)

# RPM corresponding
axes[1].plot(rpm_sequence, 'g-', linewidth=2, marker='s', markersize=4)
axes[1].set_xlabel('Sample Number')
axes[1].set_ylabel('Engine RPM')
axes[1].set_title('Corresponding Engine RPM')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('temporal_anomaly_detection.png', dpi=300)
print("✅ Saved: temporal_anomaly_detection.png")