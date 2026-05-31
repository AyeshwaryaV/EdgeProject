import matplotlib.pyplot as plt

categories = ['BLE Stack', 'Serial Buffer', 'Program Variables', 'Stack (per core)', 'Free Heap']
usage = [25, 12, 8, 8, 467]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#D4D4D4']

plt.figure(figsize=(10,6))
bars = plt.bar(categories, usage, color=colors, edgecolor='black')
plt.ylabel('Memory Usage (KB)')
plt.title('ESP32 RAM Memory Map')
plt.ylim(0, 520)

for bar, val in zip(bars, usage):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'{val} KB', ha='center', fontweight='bold')

plt.axhline(y=520, color='red', linestyle='--', label='Total RAM: 520 KB')
plt.legend()
plt.tight_layout()
plt.savefig('memory_map.png', dpi=300)
print('✅ Figure 4.10 Saved: memory_map.png')