
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 65)
print("PREDICTIVE LOGISTICS OPTIMIZATION")
print("EXPLORATORY DATA ANALYSIS")
print("=" * 65)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "logistics_modeling_data.csv"
)

FIGURES_DIR = PROJECT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load prepared dataset
df = pd.read_csv(DATA_PATH)

target = "Days for shipping (real)"

sns.set_theme(style="whitegrid")

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

# --------------------------------------------------
# CHART 1: Distribution of actual shipping days
# --------------------------------------------------

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x=target,
    color="steelblue"
)

plt.title("Distribution of Actual Shipping Days")
plt.xlabel("Actual Shipping Days")
plt.ylabel("Number of Orders")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "01_shipping_days_distribution.png",
    dpi=300
)
plt.close()

print("\nChart 1 saved.")

# --------------------------------------------------
# CHART 2: Actual vs scheduled shipping days
# --------------------------------------------------

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="Days for shipment (scheduled)",
    y=target,
    color="teal",
    errorbar=None
)

plt.title("Average Actual Shipping Days by Scheduled Days")
plt.xlabel("Scheduled Shipping Days")
plt.ylabel("Average Actual Shipping Days")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "02_actual_vs_scheduled.png",
    dpi=300
)
plt.close()

print("Chart 2 saved.")

# --------------------------------------------------
# CHART 3: Shipping duration by shipping mode
# --------------------------------------------------

plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="Shipping Mode",
    y=target,
    color="cornflowerblue",
    errorbar=None
)

plt.title("Average Shipping Duration by Shipping Mode")
plt.xlabel("Shipping Mode")
plt.ylabel("Average Actual Shipping Days")
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "03_shipping_mode_analysis.png",
    dpi=300
)
plt.close()

print("Chart 3 saved.")

# --------------------------------------------------
# CHART 4: Average shipping duration by order month
# --------------------------------------------------

monthly_shipping = (
    df.groupby("Order Month")[target]
    .mean()
    .reset_index()
)

plt.figure(figsize=(9, 5))

sns.lineplot(
    data=monthly_shipping,
    x="Order Month",
    y=target,
    marker="o",
    color="darkorange"
)

plt.title("Average Actual Shipping Duration by Order Month")
plt.xlabel("Order Month")
plt.ylabel("Average Actual Shipping Days")
plt.xticks(range(1, 13))
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "04_monthly_shipping_trend.png",
    dpi=300
)
plt.close()

print("Chart 4 saved.")

# --------------------------------------------------
# CHART 5: Correlation heatmap
# --------------------------------------------------

numeric_columns = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount Rate",
    "Order Month",
    "Order Day Of Week",
    "Order Hour",
    target
]

correlation = df[numeric_columns].corr()

plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Between Numerical Variables")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "05_correlation_heatmap.png",
    dpi=300
)
plt.close()

print("Chart 5 saved.")

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("KEY EXPLORATORY FINDINGS")
print("=" * 65)

print("\nAverage actual shipping days:")
print(round(df[target].mean(), 2))

print("\nAverage shipping duration by shipping mode:")
print(
    df.groupby("Shipping Mode")[target]
    .mean()
    .round(2)
    .sort_values()
)

print("\nAverage shipping duration by scheduled days:")
print(
    df.groupby("Days for shipment (scheduled)")[target]
    .mean()
    .round(2)
)

print("\nCorrelation with actual shipping days:")
print(
    correlation[target]
    .sort_values(ascending=False)
    .round(3)
)

print("\nAll five charts saved in the figures folder.")
print("STEP 3 COMPLETED")

