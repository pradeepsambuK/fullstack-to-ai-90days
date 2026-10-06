# Day 7/90: The environment — venvs, pinned deps, and reproducibility

## What I learned

- **Virtual environments** isolate a project's packages from your system
  Python. My honest mistake today: I forgot to activate the venv and pip
  installed scikit-learn into system Python. Twice. The script now refuses
  to run outside a venv so that mistake is loud, not silent.
- **requirements.txt is package.json for Python**: exact pinned versions,
  checked in. Pinning is what makes "works on my machine" someone else's
  machine too.
- **A notebook is a REPL you can share**: my end-to-end check re-runs days
  1 to 6 in miniature (broadcasting, groupby, correlation, cosine
  similarity, Bayes, gradient descent) with a fixed seed, so the numbers
  come out identical every time.

## The experiment

`environment.py` does three things:

1. Guards: exits with a clear message if it is not running inside a venv.
2. Pins: writes `requirements-pinned.txt` with the exact installed versions
   of numpy, pandas, matplotlib, and scikit-learn.
3. Reproduces: runs six miniature versions of the day 1 to 6 exercises.
   Loss on the tiny gradient-descent demo falls from ~44 to ~0.00 and the
   fitted slope lands at ~2.0, same as day 6.

As a full-stack dev, this day felt like finally writing a lockfile for the
whole lab: no more mystery about which versions produced which numbers.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn
python environment.py
```
