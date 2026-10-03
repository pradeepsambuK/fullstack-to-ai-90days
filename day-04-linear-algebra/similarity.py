"""Day 4/90: linear algebra intuition. Vectors, dot products, shapes.

The idea: build a dot-product similarity score between two short vectors
by hand, verify it against NumPy, then learn the hard way why "shape
mismatch" is the number one beginner error in ML.

Run:  python similarity.py   (needs numpy)
"""

import numpy as np

# Two tiny "taste" vectors: made-up scores for [action, comedy, drama].
# Think of these like feature rows for two users.
alice = [5.0, 1.0, 2.0]
bob = [4.0, 2.0, 1.0]


def dot_by_hand(a, b):
    """Dot product with nothing but a loop. Multiply pairs, add them up."""
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def cosine_by_hand(a, b):
    """Cosine similarity: how aligned two vectors point, ignoring size.

    A dot product of 20 means nothing on its own. Divide by both lengths
    and you get a number between -1 and 1 that you can actually compare.
    """
    dot = dot_by_hand(a, b)
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b)


hand_dot = dot_by_hand(alice, bob)
hand_cos = cosine_by_hand(alice, bob)

np_dot = float(np.dot(alice, bob))
np_cos = float(np.dot(alice, bob) / (np.linalg.norm(alice) * np.linalg.norm(bob)))

print(f"hand-rolled dot product : {hand_dot:.4f}")
print(f"NumPy dot product       : {np_dot:.4f}")
print(f"hand-rolled cosine      : {hand_cos:.4f}")
print(f"NumPy cosine            : {np_cos:.4f}")
assert abs(hand_dot - np_dot) < 1e-9
assert abs(hand_cos - np_cos) < 1e-9
print("hand-rolled and NumPy agree")

# The classic beginner trap: shapes.
row = np.array([1.0, 2.0, 3.0])        # shape (3,)
col = np.array([[1.0], [2.0], [3.0]])  # shape (3, 1)
print(f"\nrow shape: {row.shape}, col shape: {col.shape}")
try:
    col @ col  # (3, 1) @ (3, 1): inner dimensions 1 and 3 do not line up
    print("no error?!")
except ValueError as e:
    print(f"shape mismatch, exactly as expected: {e}")

# The full-stack bridge: matrix shapes are API contracts.
# (batch, features) in, (batch, outputs) out. A shape error is a 400
# from your tensors.
rng = np.random.default_rng(7)
batch = rng.normal(size=(4, 3))    # 4 samples, 3 features each
weights = rng.normal(size=(3, 2))  # maps 3 features to 2 outputs
out = batch @ weights
print(f"\nbatch {batch.shape} @ weights {weights.shape} -> out {out.shape}")
