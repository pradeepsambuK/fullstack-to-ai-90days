"""Day 5/90: probability intuition. Distributions, mean vs median, Bayes.

The idea: simulate 10,000 coin flips at a few biases and watch the observed
share of heads converge to the true bias as the sample grows. Then a quick
mean vs median demo on skewed data (like order prices with a few whales),
and Bayes rule by hand on a banner click example.

Run:  pip install numpy   then   python probability.py
"""

import numpy as np

rng = np.random.default_rng(5)

# Experiment 1: observed frequency converges to the true probability.
biases = [0.3, 0.5, 0.7]
sample_sizes = [10, 100, 1_000, 10_000]
print("true bias -> observed share of heads as the sample grows")
for bias in biases:
    flips = rng.random(10_000) < bias
    row = "  ".join(f"n={n:<5}: {flips[:n].mean():.3f}" for n in sample_sizes)
    print(f"bias {bias:.1f}: {row}")

# Experiment 2: mean vs median on skewed data.
# Like order prices: most are modest, a few are huge.
skewed = np.concatenate([rng.normal(loc=40, scale=5, size=990), [400.0] * 10])
print(f"\nskewed prices: mean = {skewed.mean():.2f}, median = {np.median(skewed):.2f}")
print("ten $400 orders drag the mean up; the median barely moves")

# Experiment 3: Bayes rule by hand.
# P(click | saw banner) = P(saw banner | click) * P(click) / P(saw banner)
p_click = 0.02            # P(click): 2% of visitors click at all
p_saw_given_click = 0.90  # P(saw banner | click): clickers almost always saw it
p_saw = 0.40              # P(saw banner): 40% of visitors see the banner
p_click_given_saw = p_saw_given_click * p_click / p_saw
print(f"\nP(click)              = {p_click:.3f}")
print(f"P(click | saw banner) = {p_click_given_saw:.3f}")
print("seeing the banner more than doubled the click probability")
