# Day 2/90: Pandas for data wrangling

## What I learned

- **DataFrames are the new database**: `groupby` is GROUP BY, `merge` is
  JOIN, `.query()` is WHERE. The SQL I already know, running in memory.
- **Cleaning is the job**: the raw feed had duplicate rows, missing prices,
  and eight spellings of three categories. Nothing trains until this is fixed.
- **Merges need respect**: my first merge silently multiplied rows because of
  a duplicate key in the join partner. The `assert` checking row count before
  and after the merge is now a habit I am keeping.

## The experiment

`wrangle.py` builds a messy 2000-row orders feed, cleans it, merges a
products table, and answers: which category had the highest average revenue
per order? All with vectorized pandas, zero loops.

Sample output: three categories, ~600 clean orders each, and a ranked table
of `avg_revenue_per_order` with a 100-order minimum. The merge assertion
caught my row-explosion bug before it poisoned the numbers.

## Still fuzzy

When to use `.loc` vs `.iloc` (and what SettingWithCopyWarning actually
wants from me). The error message reads like a riddle.

## Run it

```bash
pip install pandas numpy
python wrangle.py
```
