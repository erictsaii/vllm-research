import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 20.5}) 

n_values = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
ttft_values = [107.39, 96.53, 105.86, 120.93, 134.98, 142.34]

# n_values = [1.0, 0.75, 0.6, 0.4, 0.2, 0.0]
# ttft_values = [105.75, 93, 97.54, 104, 110.85, 115.36]

data = sorted(zip(n_values, ttft_values))
n_sorted = [x[0] for x in data]
ttft_sorted = [x[1] for x in data]

F = 24
T = 75

# X = 3
X = 5


ttft_at_1 = ttft_values[0]
S = ttft_at_1 - (F + T)
print(f"Calculated S: {S}") 

def calculate_ideal_ttft(n, F, X, T, S):
    term1 = max(F, X * (1 - n) * F)
    return term1 + n * T + S

n_smooth = np.linspace(0, 1, 200)
ttft_ideal = [calculate_ideal_ttft(n, F, X, T, S) for n in n_smooth]

# ---------------------------------------------------------
min_val = min(ttft_ideal)  
print(f"Normalization Base (Min Ideal TTFT): {min_val:.2f}")

ttft_exp_normalized = [x / min_val for x in ttft_sorted]
ttft_ideal_normalized = [x / min_val for x in ttft_ideal]
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.grid(True, linestyle='--', alpha=0.6)

plt.plot(n_sorted, ttft_exp_normalized, marker='o', linestyle='-', color='red', linewidth=2, label='Measured TTFT')

plt.plot(n_smooth, ttft_ideal_normalized, linestyle='-', color='blue', linewidth=2, label=f'Ideal TTFT')

plt.axvline(x=1.0 - 1.0/X, color='gray', linestyle='--', label=f'Optimal n* = {1.0 - 1.0/X}')

plt.xlabel('n')
plt.ylabel('TTFT (normalized by ideal min)')
plt.legend(loc='lower left')

plt.tight_layout()
plt.savefig("verification.png", dpi=200)
plt.show()