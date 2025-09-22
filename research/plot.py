# TTFT vs n for sequential model (compute first, then pull)
# TTFT(n) = max(1, X*(1-n)) + alpha*n, with alpha=3, X=4

import numpy as np
import matplotlib.pyplot as plt

alpha = 3
X = 4

n = np.linspace(0, 1, 1001)
ttft = np.maximum(1.0, X*(1.0 - n)) + alpha*n

# Optimal n* for the sequential model
n_star = 1.0 - 1.0/X if alpha <= X else 0.0
ttft_min = np.maximum(1.0, X*(1.0 - n_star)) + alpha*n_star

print(f"alpha = {alpha}, X = {X}")
print(f"Optimal n* = {n_star:.4f}")
print(f"Minimum TTFT = {ttft_min:.4f} (prefill forward time = 1 unit)")

plt.figure(figsize=(7,5))
plt.plot(n, ttft, label="TTFT(n) = max(1, X*(1-n)) + alpha*n", linewidth=3)
plt.axvline(n_star, linestyle="--", label=f"n* = {n_star:.3f}")
plt.scatter([n_star], [ttft_min], s=60)
plt.title(f"TTFT vs n, alpha = {alpha}, X = {X}")
plt.xlabel("n (fraction of KV transferred)")
plt.ylabel("TTFT (units of one prefill forward)")
plt.legend(loc="best")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("plot.png", dpi=200)
plt.show()
