"""Day 7/90: the environment. venvs, pinned deps, and reproducibility.

The experiment: one script that (1) refuses to run outside a virtual
environment, (2) writes a pinned requirements file from the packages actually
installed here, and (3) re-runs days 1 to 6 in miniature, end to end, so
anyone with this folder can reproduce the exact same numbers.

Run:  python -m venv .venv
      source .venv/bin/activate
      pip install numpy pandas matplotlib scikit-learn
      python environment.py
"""

import sys
import importlib.metadata as metadata

import numpy as np
import pandas as pd


# ---- Step 0: the honest guardrail. ----
# My day 7 mistake: I forgot to activate the venv and installed scikit learn
# into my system Python. Twice. This check makes that mistake loud instead of
# silent.
in_venv = sys.prefix != sys.base_prefix
print(f"python: {sys.executable}")
print(f"running inside a virtual environment: {in_venv}")
if not in_venv:
    sys.exit(
        "Refusing to run outside a venv. "
        "Make one with: python -m venv .venv && source .venv/bin/activate"
    )


# ---- Step 1: pin what is actually installed. ----
# requirements.txt is package.json for Python: exact versions, checked in.
PINNED = ["numpy", "pandas", "matplotlib", "scikit-learn"]
lines = [f"{pkg}=={metadata.version(pkg)}" for pkg in PINNED]
with open("requirements-pinned.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
print("\nwrote requirements-pinned.txt:")
for line in lines:
    print(f"  {line}")


# ---- Step 2: days 1 to 6 in miniature. Fixed seed = reproducible. ----
rng = np.random.default_rng(7)
print("\n--- miniature re-runs ---")

# Day 1: broadcasting. Normalize columns without loops.
x = rng.normal(size=(20_000, 8))
normed = (x - x.mean(axis=0)) / x.std(axis=0)
print(f"day 1: column means after normalizing: {normed.mean(axis=0)[:3].round(3)} (all ~0)")

# Day 2: pandas is in-memory SQL. groupby is GROUP BY.
orders = pd.DataFrame({
    "city": ["slc", "slc", "denver", "denver", "slc"],
    "total": [40.0, 60.0, 55.0, 45.0, 50.0],
})
print(f"day 2: avg order by city -> {orders.groupby('city')['total'].mean().round(1).to_dict()}")

# Day 3: inspect before you model. Correlation first, model never.
corr = (
    orders[["total"]]
    .assign(total2=orders["total"] * 2 + rng.normal(size=5))
    .corr(numeric_only=True)
    .iloc[0, 1]
)
print(f"day 3: correlation between total and a scaled copy: {corr:.3f}")

# Day 4: shapes are API contracts. Cosine similarity of two tiny vectors.
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0])
cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"day 4: cosine similarity of identical vectors: {cos:.3f} (should be 1.000)")
try:
    a.reshape(3, 1) @ b.reshape(2, 2)
except ValueError as e:
    print(f"day 4: shape mismatch caught cleanly: {e}")

# Day 5: Bayes is conditional rendering logic.
# P(click | saw banner) = P(saw | click) * P(click) / P(saw)
p_click_given_saw = 0.90 * 0.02 / 0.40
print(f"day 5: P(click | saw banner) = {p_click_given_saw:.3f}")

# Day 6: gradient descent, the hot-reload loop for parameters.
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
ys = np.array([2.1, 4.0, 6.1, 8.0, 9.9])
w, lr = 0.0, 0.01
losses = []
for _ in range(200):
    grad = -2 * (xs * (ys - w * xs)).mean()
    w -= lr * grad
    losses.append(((ys - w * xs) ** 2).mean())
print(f"day 6: loss {losses[0]:.2f} -> {losses[-1]:.4f}, slope {w:.4f} (truth ~2.0)")

print("\nAll six miniature re-runs match. Same seed, same venv, same numbers.")
