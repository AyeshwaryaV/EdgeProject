import matplotlib.pyplot as plt
import numpy as np

categories = ['Latency', 'Privacy', 'Offline Ops', 'Cost', 'Data Usage', 'Internet Required']
edge_scores = [100, 100, 100, 100, 100, 100]
cloud_scores = [1, 20, 0, 30, 20, 0]

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
edge_scores += edge_scores[:1]
cloud_scores += cloud_scores[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
ax.plot(angles, edge_scores, 'o-', linewidth=2, label='Edge (This Work)', color='blue')
ax.fill(angles, edge_scores, alpha=0.25, color='blue')
ax.plot(angles, cloud_scores, 'o-', linewidth=2, label='Cloud-Based', color='red')
ax.fill(angles, cloud_scores, alpha=0.25, color='red')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 100)
ax.set_title('Edge vs Cloud Architecture Comparison', size=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig('radar_chart.png', dpi=300)
print('✅ Figure 4.12 Saved: radar_chart.png')