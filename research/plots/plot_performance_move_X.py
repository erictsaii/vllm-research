import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 15}) 

categories = ["X=3", "X=4", "X=5"]
original = np.array([285, 285, 285])
optimal = np.array([130, 172.6, 245])

# original = np.array([741, 741, 741])
# optimal = np.array([480, 550, 600])

orig_norm = np.ones_like(original, dtype=float)  # baseline 1.0
opt_norm = optimal / original

x = np.arange(len(categories))
width = 0.36

fig, ax = plt.subplots(figsize=(7.5, 4.8))

rects1 = ax.bar(x - width/2, orig_norm, width, label="vllm", color="#CACACA",
                edgecolor='black', linewidth=1)
rects2 = ax.bar(x + width/2, opt_norm, width, label="NIKA", color="#4C68F2",
                edgecolor='black', linewidth=1)

ax.set_ylabel("Normalized TTFT")
ax.set_xticks(x, categories)
ax.set_ylim(0, 1.27)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
ax.legend()

ax.bar_label(rects1, labels=[f"{v:.2f}" if not np.isclose(v, 1.0) else "" for v in orig_norm], padding=3)
ax.bar_label(rects2, labels=[f"{v:.2f}" if not np.isclose(v, 1.0) else "" for v in opt_norm], padding=3)
# ----------------------------------------------

ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
out_path = "performance_move_X.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.show()