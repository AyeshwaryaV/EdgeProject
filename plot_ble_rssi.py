import matplotlib.pyplot as plt

distances = [1, 3, 5, 8, 10, 15]
rssi_los = [-40, -52, -60, -70, -75, -82]
rssi_chassis = [-42, -55, -65, -78, -85, -95]

plt.figure(figsize=(10,6))
plt.plot(distances, rssi_los, 'bo-', linewidth=2, markersize=8, label='Line of Sight')
plt.plot(distances, rssi_chassis, 'rs-', linewidth=2, markersize=8, label='Through Chassis')
plt.xlabel('Distance (meters)')
plt.ylabel('RSSI (dBm)')
plt.title('BLE Signal Strength vs Distance')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-100, -30)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('ble_rssi_plot.png', dpi=300)
print('✅ Figure 4.11 Saved: ble_rssi_plot.png')