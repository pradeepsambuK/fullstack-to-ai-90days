"""
Day 8/90: The ML Workflow - Data, Train, Evaluate, Iterate.

Experiment: run a full ML pipeline on a spam classifier, with each stage
labeled, and show where each Day 1-7 skill slots in:
    Day 1 numpy/broadcasting  -> CLEAN: normalize count features
    Day 2 pandas/groupby      -> EXPLORE: compare word rates, spam vs ham
    Day 3 visualization       -> EXPLORE: ASCII bar chart of spammy words
    Day 4 linear algebra      -> TRAIN: log-probability dot products
    Day 5 Bayes               -> TRAIN: the naive Bayes classifier itself
    Day 6 gradient descent    -> ITERATE: tuning the smoothing alpha by hand
    Day 7 environments        -> everything: fixed seed, pinned deps

The honest-vs-cheating moment: score the model on its training data
(cheating, looks great) and then on a held-out test set (the real number).

Run:  pip install numpy   then   python spam_workflow.py
"""
import numpy as np

rng = np.random.default_rng(8)
N = 1000

# ------------------------------ COLLECT --------------------------------
# Synthetic emails. Binary word features + two count features + a label.
# Spammy words ("free", "win", "urgent") lean spam; neutral words
# ("invoice", "meeting", "hello") lean ham. "hello" appears everywhere:
# a constant column, which is useless and gets dropped in CLEAN.
words = ["free", "win", "urgent", "invoice", "meeting", "hello"]
spam_word_p = np.array([0.65, 0.55, 0.45, 0.05, 0.05, 0.80])
ham_word_p = np.array([0.02, 0.01, 0.03, 0.25, 0.30, 0.80])

labels = rng.integers(0, 2, size=N)              # 1 = spam, 0 = ham
X_bin = np.where(
    (labels == 1)[:, None],
    rng.random((N, 6)) < spam_word_p,
    rng.random((N, 6)) < ham_word_p,
).astype(float)
exclams = np.where(labels == 1, rng.poisson(4, N), rng.poisson(1, N)).astype(float)
links = np.where(labels == 1, rng.poisson(2, N), rng.poisson(0, N)).astype(float)

X = np.column_stack([X_bin, exclams, links])
feat_names = words + ["exclamations", "links"]
print(f"[collect] {N} emails, {X.shape[1]} features, spam rate {labels.mean():.1%}")

# ------------------------------- CLEAN ---------------------------------
# Drop constant columns ("hello" is in every email, tells us nothing) and
# normalize the count features with broadcasting, straight out of Day 1.
const_cols = np.where(X.std(axis=0) == 0)[0]
X = np.delete(X, const_cols, axis=1)
feat_names = [f for i, f in enumerate(feat_names) if i not in const_cols]
count_idx = [len(feat_names) - 2, len(feat_names) - 1]  # last two are counts
X[:, count_idx] = (X[:, count_idx] - X[:, count_idx].mean(axis=0)) / (
    X[:, count_idx].std(axis=0) + 1e-9
)
print(f"[clean] dropped constant column 'hello'; normalized {count_idx} via broadcasting")

# ------------------------------ EXPLORE --------------------------------
# Day 2 groupby and Day 3 visualization, the scrappy numpy edition.
for word, i in [("free", 0), ("invoice", 3)]:
    print(f"[explore] P('{word}'|spam)={X[labels==1, i].mean():.2f} "
          f"vs P('{word}'|ham)={X[labels==0, i].mean():.2f}")
print("[explore] spammiest word indicators (spam rate, ASCII chart):")
for i, name in enumerate(feat_names[:5]):
    rate = X[labels == 1, i].mean()
    print(f"  {name:>10} {'#' * int(rate * 40):<40} {rate:.2f}")

# ------------------------------- SPLIT ---------------------------------
# Day 11 preview: hold out 20% before training. Never train on the test set.
idx = rng.permutation(N)
cut = int(0.8 * N)
X_train, X_test = X[idx[:cut]], X[idx[cut:]]
y_train, y_test = labels[idx[:cut]], labels[idx[cut:]]
print(f"[split] train={len(y_train)}, test={len(y_test)}")

# ------------------------------- TRAIN ---------------------------------
# Bernoulli naive Bayes (Day 5), computed with matrix dot products (Day 4),
# tuned by hand instead of gradient descent (Day 6):
# I just grid-search the Laplace smoothing alpha and keep the best.
def train_nb(Xt, yt, alpha, n_bin=5):
    log_prior = np.log(np.bincount(yt, minlength=2) / len(yt))
    # keep binary word features binary; binarize the count features
    Xb = np.where(Xt[:, n_bin:] > 0, 1.0, 0.0)
    Xb = np.column_stack([Xt[:, :n_bin], Xb])
    probs = (Xb[yt == 1].sum(axis=0) + alpha) / ((yt == 1).sum() + 2 * alpha)
    probs0 = (Xb[yt == 0].sum(axis=0) + alpha) / ((yt == 0).sum() + 2 * alpha)
    return log_prior, np.log(probs), np.log(1 - probs), np.log(probs0), np.log(1 - probs0)

def predict(model, Xe, n_bin=5):
    log_prior, lp1, lq1, lp0, lq0 = model
    Xb = np.column_stack([Xe[:, :n_bin], np.where(Xe[:, n_bin:] > 0, 1.0, 0.0)])
    score1 = log_prior[1] + Xb @ lp1 + (1 - Xb) @ lq1
    score0 = log_prior[0] + Xb @ lp0 + (1 - Xb) @ lq0
    return (score1 > score0).astype(int)

def accuracy(model, Xe, ye):
    return (predict(model, Xe) == ye).mean()

best = max(((a, accuracy(train_nb(X_train, y_train, a), X_train, y_train))
            for a in [0.1, 0.5, 1.0, 2.0]), key=lambda t: t[1])
alpha = best[0]
model = train_nb(X_train, y_train, alpha)
print(f"[train] naive Bayes trained, best smoothing alpha={alpha} (the iterate step)")

# ----------------------------- EVALUATE --------------------------------
train_acc = accuracy(model, X_train, y_train)
test_acc = accuracy(model, X_test, y_test)
print(f"[evaluate] CHEATING (score on train data): {train_acc:.1%}")
print(f"[evaluate] HONEST   (score on test data):  {test_acc:.1%}")
pred = predict(model, X_test)
tp = int(((pred == 1) & (y_test == 1)).sum())
fp = int(((pred == 1) & (y_test == 0)).sum())
tn = int(((pred == 0) & (y_test == 0)).sum())
fn = int(((pred == 0) & (y_test == 1)).sum())
print(f"[evaluate] test confusion: TP={tp} FP={fp} TN={tn} FN={fn}")

# ------------------------------ ITERATE --------------------------------
# Iterate the loop: drop the weakest feature (lowest spam signal) and
# retrain. Did the honest score move? Print it, whatever it says.
signal = np.abs(X_train[y_train == 1, :5].mean(axis=0)
                - X_train[y_train == 0, :5].mean(axis=0))
weak = int(np.argmin(signal))
print(f"[iterate] dropping weakest word feature '{feat_names[weak]}', retraining...")
keep = [i for i in range(5)] + [5, 6]
keep.remove(weak)
X2_train, X2_test = X_train[:, keep], X_test[:, keep]
m2 = train_nb(X2_train, y_train, alpha, n_bin=4)
acc2 = (predict(m2, X2_test, n_bin=4) == y_test).mean()
print(f"[iterate] test accuracy after dropping it: {acc2:.1%} "
      f"(was {test_acc:.1%}, delta {acc2 - test_acc:+.1%})")
print("pipeline done: collect -> clean -> split -> train -> evaluate -> iterate")
