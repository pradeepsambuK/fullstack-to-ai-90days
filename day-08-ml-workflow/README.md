# Day 8/90: The ML Workflow — Data, Train, Evaluate, Iterate

## What I learned

- Every ML project runs the same loop: collect → clean → split → train →
  evaluate → iterate. It is the SDLC I already know, just with probabilistic
  outputs instead of unit tests.
- My whole first week slots into the loop: NumPy broadcasting in **clean**,
  pandas-style groupby in **explore**, Bayes in **train**, hand-tuned search
  in **iterate**, and a pinned env holding the whole pipeline steady.
- Scoring on the training data is cheating: it looked near perfect (98.5%).
  The held-out test set told the truth (97.5%).
- Iterating is not the same as improving: dropping a weak-looking word
  feature cost 6 points, because "invoice" was quietly carrying the ham
  signal.

## The experiment

`spam_workflow.py` runs a hand-rolled Bernoulli naive Bayes spam classifier
(pure NumPy, log probabilities with dot products) through each pipeline
stage on 1000 synthetic emails, printing every stage:

| stage | score |
|-------|-------|
| cheating (score on train data) | 98.5% |
| honest (score on test data) | 97.5% |
| after dropping weakest feature | 91.5% (delta −6.0%) |

The bug I actually hit: binarizing normalized counts left raw negative
values in the feature matrix, which produced negative "probabilities" and
NaNs. Fixed by mapping them to 0. Classic "the math is right but the data
isn't" moment.

## Still fuzzy

How you know when to stop iterating. The loop has no natural exit. In web
dev, the ticket closes. In ML, the scoreboard just keeps moving.

## Run it

```bash
pip install numpy
python spam_workflow.py
```
