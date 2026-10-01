"""
Day 2/90: Pandas for data wrangling.

Experiment: build a messy orders dataset (dupes, missing prices,
inconsistent category casing, extra whitespace), clean it with pandas,
merge in a products table, and answer a real question:
which category had the highest average revenue per order?

Run:  pip install pandas numpy   then   python wrangle.py
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(2)

# --- a messy raw orders feed, like something pasted from five spreadsheets ---
categories = ["  Electronics", "ELECTRONICS", "electronics ", "Books",
              " books", "BOOKS", "Clothing", "clothing"]
rows = []
for i in range(2000):
    rows.append({
        "order_id": i,
        "product": f"P{i % 40:03d}",
        "category": categories[rng.integers(len(categories))],
        "qty": int(rng.integers(1, 5)),
        "unit_price": float(np.round(rng.normal(40, 15), 2)) if rng.random() > 0.08 else None,
    })
orders = pd.DataFrame(rows)
orders = pd.concat([orders, orders.sample(50, random_state=9)])  # duplicate rows sneak in
print("raw:", orders.shape, "| null prices:", int(orders["unit_price"].isna().sum()),
      "| dupes:", int(orders.duplicated().sum()))

# --- clean it ---
clean = (orders
         .drop_duplicates()                      # WHERE order_id IS DISTINCT
         .copy())
clean["category"] = clean["category"].str.strip().str.lower()   # normalize the mess
median_price = clean["unit_price"].median()
clean["unit_price"] = clean["unit_price"].fillna(median_price)  # honest imputation, not zero
print("clean:", clean.shape)

# --- products reference table, the JOIN partner ---
products = pd.DataFrame({
    "product": [f"P{i:03d}" for i in range(40)],
    "name": [f"item-{i:03d}" for i in range(40)],
    "weight_kg": np.round(rng.uniform(0.1, 5.0, 40), 2),
})

# gotcha I actually hit: duplicate keys in the join partner blow up row counts
enriched = clean.merge(products, on="product", how="left")
assert len(enriched) == len(clean), "row count changed after merge!"
enriched["revenue"] = enriched["qty"] * enriched["unit_price"]

# --- answer the question: highest average revenue per order by category ---
answer = (enriched
          .groupby("category")
          .agg(orders=("order_id", "count"),
               total_revenue=("revenue", "sum"),
               avg_revenue_per_order=("revenue", "mean"))
          .round(2)
          .query("orders >= 100")            # WHERE clause, in pandas
          .sort_values("avg_revenue_per_order", ascending=False))

print()
print("average revenue per order by category (min 100 orders):")
print(answer)
print()
winner = answer.index[0]
print(f"answer: '{winner}' wins at ${answer.iloc[0]['avg_revenue_per_order']}/order")
