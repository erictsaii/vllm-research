import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
components = ['KV Cache Transfer', 'Model Forward', 'Sampling']
percentages = np.array([83.5, 16, 0.5])

bar_width = 0.3
bar_position = 0

# --- Colors ---
colors = {
    'Model Forward': 'cornflowerblue',
    'KV Cache Transfer': 'firebrick',
    'Sampling': 'seagreen'
}

# --- Plot ---
fig, ax = plt.subplots(figsize=(4, 7))
bottom = 0

bars = []
for component, percent in zip(components, percentages):
    bar = ax.bar(
        bar_position,
        percent,
        width=bar_width,
        bottom=bottom,
        label=component,
        color=colors[component],
        edgecolor='black',   # 黑色外框
        linewidth=1        # 外框寬度
    )
    bars.append(bar)
    bottom += percent

# --- Axis & Legend ---
ax.set_ylabel('Percentage of Total TTFT (%)', fontsize=12)
ax.set_ylim(0, 100)
ax.set_xlim(-0.3, 0.3)
ax.set_xticks([])

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    frameon=False,
    fontsize=11
)

ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)
plt.tight_layout(rect=[0, 0.1, 1, 1])

plt.savefig('ttft_baseline_breakdown.png', dpi=300, bbox_inches='tight')
plt.show()
