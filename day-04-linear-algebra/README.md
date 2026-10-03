# Day 4/90: Linear algebra intuition — vectors, dot products, shapes

## What I learned

- **Vectors** are just lists of numbers with a direction: a row of features
  for one sample, like `[5, 1, 2]` for a user's made-up movie-taste scores.
- **Dot product** measures how much two vectors agree: multiply matching
  entries, add them up. Big shared entries → big number.
- **Cosine similarity** is the dot product normalized to [-1, 1], so scores
  are actually comparable. Same direction → close to 1, unrelated → near 0.
- **Shapes are API contracts.** `(batch, features)` in, `(batch, outputs)`
  out. A `(3,)` vector and a `(3, 1)` matrix hold the same three numbers but
  are different objects, and multiplying the wrong pair gives the classic
  `ValueError: matmul` shape mismatch. As a full-stack dev, I read that as a
  400 Bad Request from my tensors.

## The experiment

`similarity.py` scores how similar two toy "taste" vectors are, twice:

| step | method |
|------|--------|
| 1 | dot product + cosine similarity written by hand (plain loop) |
| 2 | the same two numbers via NumPy, with `assert`s that they match |
| 3 | a deliberate `(3, 1) @ (3, 1)` to watch the shape mismatch fire |
| 4 | a `(4, 3) @ (3, 2)` "mini model" showing the (batch, features) → (batch, outputs) contract |

Hand-rolled and NumPy agreed to 9 decimal places. The mismatch demo fired
exactly the error every beginner meets.

## Still fuzzy

Why `(3,)` and `(3, 1)` are different things. Sitting with that one
(carried over from Day 1 — it finally bit me today).

## How to run

```bash
cd day-04-linear-algebra
pip install numpy
python similarity.py
```
