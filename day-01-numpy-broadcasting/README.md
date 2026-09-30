# Day 1/90: NumPy, vectors, and broadcasting

## What I learned

- **ndarrays**: n-dimensional arrays, the core data structure behind every
  model. Everything in ML is arrays of numbers.
- **Broadcasting**: NumPy's way of doing math on arrays of different shapes
  with zero loops. A vector of shape `(8,)` stretches across every row of a
  `(200000, 8)` matrix.
- **Shapes are API contracts**: `(batch, features)` in, `(batch, outputs)`
  out. My first shape-mismatch error taught me more than the docs did.

## The experiment

`broadcasting.py` normalizes every column of a 200k x 8 dataset two ways and
times both:

| method | time | result |
|--------|------|--------|
| pure NumPy (broadcasting) | ~0.01s | same |
| plain Python loop | ~1s | same |

NumPy was about 100x faster with the identical result. As a full-stack dev,
it reminded me of swapping raw DOM manipulation for a virtual DOM: the work
moved to optimized C under the hood.

## Still fuzzy

Why `(3,)` and `(3, 1)` are different things. Sitting with that one.

## Run it

```bash
pip install numpy
python broadcasting.py
```
