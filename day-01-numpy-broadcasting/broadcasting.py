"""
Day 1/90: NumPy, vectors, and broadcasting.

Experiment: normalize every column of a synthetic dataset two ways,
pure NumPy (vectorized, with broadcasting) vs a plain Python loop,
and time both.

Run:  pip install numpy   then   python broadcasting.py
"""
import time

import numpy as np

rng = np.random.default_rng(7)
data = rng.normal(loc=50, scale=10, size=(200_000, 8))  # 200k rows, 8 cols


def normalize_numpy(x):
    # broadcasting: mean/std have shape (8,) and stretch across every row,
    # no loops written by me
    return (x - x.mean(axis=0)) / x.std(axis=0)


def normalize_loop(x):
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        col = x[:, j]
        out[:, j] = (col - col.mean()) / col.std()
    return out


timings = {}
for name, fn in [("numpy", normalize_numpy), ("python loop", normalize_loop)]:
    start = time.perf_counter()
    result = fn(data)
    elapsed = time.perf_counter() - start
    timings[name] = elapsed
    print(f"{name:>12}: {elapsed:.3f}s")

print("same result:", np.allclose(normalize_numpy(data), normalize_loop(data)))
print(f"numpy was ~{timings['python loop'] / timings['numpy']:.0f}x faster")
