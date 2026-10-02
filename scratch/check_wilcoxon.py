import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("results/05_robustness/enhanced_sensor_dropout_50runs_raw.csv")
print("Raw data columns:", df.columns.tolist())

for rate in [0.10, 0.25, 0.50]:
    sub = df[df["dropout_rate"] == rate]
    buffers = sub["buffer_vs_imp"].values
    n = len(buffers)
    n_pos = np.sum(buffers > 0)
    n_neg = np.sum(buffers < 0)
    n_zero = np.sum(buffers == 0)
    w_stat, p_val = stats.wilcoxon(buffers)
    print(f"Rate {rate*100:.0f}%: N={n}, Pos={n_pos}, Neg={n_neg}, Zero={n_zero}")
    print(f"  Min buffer: {buffers.min():.4f}, Max buffer: {buffers.max():.4f}")
    print(f"  Wilcoxon stat: {w_stat}, p-value: {p_val:.8e} ({p_val})")

# Let's also check theoretical minimum p-value for N=50
# If all ranks have the same sign, W = 0 (or sum(1..50) = 1275)
# In scipy.stats.wilcoxon, asymptotic normal approx is used for N >= 50 or exact distribution:
print("\nTheoretical calculation:")
# Scipy wilcoxon behavior for N=50 with all positive differences:
test_arr = np.ones(50)
print("All +1 test array:", stats.wilcoxon(test_arr))
test_arr_diff = np.arange(1, 51)
print("Arbitrary strictly positive ranks:", stats.wilcoxon(test_arr_diff))
