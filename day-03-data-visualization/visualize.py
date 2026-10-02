"""
Day 3/90: Visualizing data before modeling.

Experiment: take the cleaned Day 2 orders dataset and LOOK at it before
training anything. Plot the unit_price distribution, a scatter of price vs
revenue (colored by category), and a correlation heatmap of the numeric
features. Write down three observations.

Bridge: like checking the network tab before debugging the backend --
inspect the raw material before blaming the logic.

Run:  pip install pandas numpy matplotlib   then   python visualize.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: save figures, don't try to open windows
import matplotlib.pyplot as plt

rng = np.random.default_rng(2)

# --- rebuild the same cleaned dataset as day 2 (self-contained on purpose) ---
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
orders = pd.concat([orders, orders.sample(50, random_state=9)])
clean = orders.drop_duplicates().copy()
clean["category"] = clean["category"].str.strip().str.lower()
median_price = clean["unit_price"].median()
was_missing = clean["unit_price"].isna()
clean["unit_price"] = clean["unit_price"].fillna(median_price)
products = pd.DataFrame({
    "product": [f"P{i:03d}" for i in range(40)],
    "weight_kg": np.round(rng.uniform(0.1, 5.0, 40), 2),
})
df = clean.merge(products, on="product", how="left")
df["revenue"] = df["qty"] * df["unit_price"]
n_imputed = int(was_missing.sum())
print(f"clean rows: {len(df)} | prices imputed with median ${median_price:.2f}: {n_imputed}")

# --- observation 1: the price distribution ---
fig, ax = plt.subplots()
ax.hist(df["unit_price"], bins=40, color="#0E7C7B", edgecolor="white")
ax.axvline(median_price, color="#E8A838", linestyle="--",
           label=f"median (imputed) ${median_price:.2f}")
ax.set_xlabel("unit price ($)")
ax.set_ylabel("orders")
ax.set_title("Distribution of unit_price")
ax.legend()
fig.tight_layout()
fig.savefig("price_distribution.png")
print("saved price_distribution.png")

skew = df["unit_price"].skew()
mode_share = (df["unit_price"] == median_price).mean()
print(f"price skew: {skew:.2f} (right tail) | exactly at median: {mode_share:.1%}")

# --- observation 2: price vs revenue scatter, colored by category ---
fig, ax = plt.subplots()
for cat in sorted(df["category"].unique()):
    sub = df[df["category"] == cat]
    ax.scatter(sub["unit_price"], sub["revenue"], s=12, alpha=0.5, label=cat)
ax.set_xlabel("unit price ($)")
ax.set_ylabel("revenue ($)")
ax.set_title("Revenue vs unit price (fan = qty levels 1 to 4)")
ax.legend()
fig.tight_layout()
fig.savefig("price_vs_revenue.png")
print("saved price_vs_revenue.png")

top = df.nlargest(3, "revenue")[["order_id", "category", "qty", "unit_price", "revenue"]]
print("top 3 revenue orders:")
print(top.to_string(index=False))

# --- observation 3: correlation heatmap of numeric features ---
num = df[["qty", "unit_price", "weight_kg", "revenue"]]
corr = num.corr()
fig, ax = plt.subplots()
im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdYlGn")
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=20, ha="right")
ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
ax.set_title("Correlation heatmap: numeric features")
fig.colorbar(im, ax=ax, label="correlation")
fig.tight_layout()
fig.savefig("correlation_heatmap.png")
print("saved correlation_heatmap.png")

print()
print("THREE OBSERVATIONS")
print("1.", f"The price histogram is roughly symmetric (skew {skew:.2f}) but has a "
      f"visible spike at ${median_price:.2f}: {n_imputed} imputed prices all landed "
      "on the median. Yesterday's imputation choice shows up in today's plot.")
print("2.", "The scatter fans into 4 bands: each band is a qty level (1-4), "
      "since revenue = qty x price. A table hides this; the plot shows it instantly.")
print("3.", f"revenue correlates {corr.loc['revenue', 'unit_price']:.2f} with price and "
      f"{corr.loc['revenue', 'qty']:.2f} with qty (it's literally their product), "
      f"while product weight sits at {corr.loc['revenue', 'weight_kg']:.2f}: "
      "derived features light up, independent ones stay dark.")
