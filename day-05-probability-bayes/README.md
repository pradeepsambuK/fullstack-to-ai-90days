# Day 5/90: Probability intuition — distributions and Bayes

## What I learned

- **Model outputs are probabilities, not certainties.** A classifier saying
  0.7 means "my best bet", not "fact".
- **Observed frequencies converge.** 10,000 coin flips land much closer to the
  true bias than 10 do. Noise washes out with sample size.
- **Mean vs median.** On skewed data (order prices with a few whales) the mean
  gets dragged around; the median stays honest.
- **Bayes rule**: P(A|B) = P(B|A) * P(A) / P(B). Evidence updates the belief.
  I kept reading it backwards at first — the order matters.

## The experiment

`probability.py` runs three small demos:

| step | what it does |
|------|--------------|
| 1 | flips 10,000 coins at biases 0.3, 0.5, 0.7; prints observed heads share at n = 10, 100, 1000, 10000 |
| 2 | builds a skewed price list (990 normal prices + ten $400 orders); compares mean vs median |
| 3 | computes P(click \| saw banner) from prior, likelihood, and marginal, by hand |

Convergence held at every bias. The mean landed ~44 while the median sat
~40, exactly the skew I built in. And the banner example: a 2% base click
rate became 4.5% after conditioning on seeing the banner.

## Still fuzzy

Uniform vs normal: my gut still expects every distribution to look
bell-shaped. Need to sit with flat distributions a bit more.

## How to run

```bash
cd day-05-probability-bayes
pip install numpy
python probability.py
```
