# Day 6: Calculus Intuition — Derivatives and Gradient Descent

## What I learned
A derivative is just a slope: how fast the loss changes when I nudge a
parameter. Gradient descent is the mechanical version of that idea — measure
the slope of the loss, take a small step downhill, repeat. Training a model
is rolling a ball downhill until the error stops shrinking.

## The experiment
`gradient_descent.py` fits a one-parameter model (y = w * x) to 5 points by
hand. No autograd, no frameworks: the MSE gradient is derived on paper and
coded directly, then w is updated step by step.

Three learning rates tell the real story:

- **lr = 0.05 (good):** loss fell from 44.18 to 0.02 in about 10 steps, and w
  settled at 2.0036 — essentially the true slope of the data.
- **lr = 0.5 (too big, my first attempt):** the steps overshot the minimum
  every time and the loss exploded into the quintillions. Bigger steps are
  not faster learning; they are jumping over the valley and landing higher
  on the other side.
- **lr = 0.0005 (too small):** 100 steps barely moved the loss from 44.18
  down to 4.86. Safe, but effectively standing still.

The takeaway I will remember: the learning rate is the most load-bearing
knob in ML, and picking it is a feel thing you only earn by blowing up a
few training runs.

## How to run
```bash
python gradient_descent.py
```
Requires only NumPy. Output prints the loss curve for all three cases.
