# Day 3/90: Visualizing data before modeling

## What I learned

- **Plots are the data debugger**: the histogram showed me my own
  imputation (163 missing prices all stacked on the median $40.32) in a way
  no `.describe()` ever would. I did that yesterday and only saw it today.
- **Scatter beats tables for structure**: plotting revenue against price
  fans into 4 clean bands, one per quantity level, because
  revenue = qty x price. The structure was sitting in the data the whole
  time; the plot just made it visible.
- **Correlation heatmaps rank features fast**: revenue correlates 0.60
  with price and 0.75 with quantity, while product weight sits at 0.03.
  Derived columns light up, independent ones stay dark. One glance tells
  you where the signal lives.

## The experiment

`visualize.py` rebuilds the cleaned orders dataset from Day 2 (same seed,
self-contained) and produces three figures: `price_distribution.png`,
`price_vs_revenue.png`, and `correlation_heatmap.png`. It then prints three
written observations derived from the plots, not the tables.

Sample output: 2000 clean rows, skew 0.06 (roughly symmetric prices), the
4-band fan in the scatter, and the top order at $346.60 (electronics,
qty 4 at $86.65).

## Still fuzzy

When to reach for a log scale on skewed distributions, and which plot type
answers which question. I picked histograms and scatters by instinct today,
not by rule.

## Run it

```bash
pip install pandas numpy matplotlib
python visualize.py
```
