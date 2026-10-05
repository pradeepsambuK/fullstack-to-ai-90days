"""
Day 6: Calculus intuition — derivatives and gradient descent.

Training = rolling downhill to the lowest error. Here we fit a one-parameter
model (y = w * x, no intercept) to 5 points by hand: compute the gradient of
the mean-squared-error loss ourselves and take small steps downhill.

Run: python gradient_descent.py
"""
import numpy as np

# Five hand-picked points, roughly on the line y = 2 * x with a little noise.
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
ys = np.array([2.1, 3.9, 6.2, 7.8, 10.1])

def predict(w, x):
    return w * x

def mse_loss(w):
    return float(np.mean((predict(w, xs) - ys) ** 2))

def mse_gradient(w):
    # d/dw of mean((w*x - y)^2) = mean(2 * (w*x - y) * x)
    return float(np.mean(2 * (predict(w, xs) - ys) * xs))

def train(w_start, learning_rate, steps, label):
    w = w_start
    print(f"\n--- {label} (start w={w_start}, lr={learning_rate}) ---")
    print(f"step    w        loss")
    for step in range(steps + 1):
        loss = mse_loss(w)
        if step % 10 == 0 or step == steps:
            print(f"{step:<6d}  {w:7.4f}  {loss:.6f}")
        if step < steps:
            w = w - learning_rate * mse_gradient(w)
    return w

# Case 1: sensible learning rate — the loss curve drops.
train(w_start=0.0, learning_rate=0.05, steps=100, label="lr = 0.05 (good)")

# Case 2: my first attempt — learning rate way too big.
# The steps overshoot the minimum and the loss explodes. Classic beginner
# mistake: "bigger steps = faster learning" is wrong when you jump over
# the valley and land higher on the other side.
train(w_start=0.0, learning_rate=0.5, steps=30, label="lr = 0.5 (too big)")

# Case 3: learning rate too small — it crawls and barely moves.
train(w_start=0.0, learning_rate=0.0005, steps=100, label="lr = 0.0005 (too small)")
