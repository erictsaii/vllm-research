# Create a normalized grouped bar chart for TTFT data
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 15}) 


X = 5

categories = ["B=5", "B=4", "B=3"]
original = np.array([76 , 87.66, 100.8])
optimal = np.array([66 , 53.63, 42.24])

# categories = ["B=5", "B=4", "B=3"]
# original = np.array([244.35 , 280, 342.45])
# optimal = np.array([212 , 184.01, 150])

# original = np.array([741, 741, 741])
# optimal = np.array([600, 540, 489])

orig_norm = np.ones_like(original, dtype=float)  # baseline 1.0
opt_norm = optimal / original

x = np.arange(len(categories))
width = 0.36

fig, ax = plt.subplots(figsize=(7.5, 4.8))

# 每個 X 放兩根柱：左 original、右 optimal
rects1 = ax.bar(x - width/2, orig_norm, width, label="vllm", color="#CACACA",
                edgecolor='black', linewidth=1)
rects2 = ax.bar(x + width/2, opt_norm, width, label="NIKA", color="#4C68F2",
                edgecolor='black', linewidth=1)
    
# ax.set_title(f"TTFT reduction (X = {X})")
ax.set_ylabel("Normalized TTFT")
ax.set_xticks(x, categories)
ax.set_ylim(0, 1.27)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]) 
ax.legend()

ax.bar_label(rects1, labels=[f"{v:.2f}" if not np.isclose(v, 1.0) else "" for v in orig_norm], padding=3)
ax.bar_label(rects2, labels=[f"{v:.2f}" if not np.isclose(v, 1.0) else "" for v in opt_norm], padding=3)


ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
out_path = "performance_move_T.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved to: {out_path}")
