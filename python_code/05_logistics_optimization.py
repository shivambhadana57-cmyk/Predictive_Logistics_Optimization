
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 65)
print("PREDICTIVE LOGISTICS OPTIMIZATION")
print("LOGISTICS PERFORMANCE AND OPTIMIZATION ANALYSIS")
print("=" * 65)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "logistics_modeling_data.csv"
)

PREDICTION_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "validated_test_predictions.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
FIGURE_DIR = PROJECT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
print("\nLoading prepared data and test predictions...")

df = pd.read_csv(DATA_PATH)
predictions = pd.read_csv(PREDICTION_PATH)

# Recreate the original test split to align predictions
from sklearn.model_selection import train_test_split

target = "Days for shipping (real)"

X = df.drop(columns=[target])
y = df[target]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Attach test features to predictions
analysis = X_test.reset_index(drop=True).copy()

analysis["Actual Shipping Days"] = predictions[
    "Actual Shipping Days"
].to_numpy()

analysis["Predicted Shipping Days"] = predictions[
    "Predicted Shipping Days"
].to_numpy()

# Calculate shipping duration difference
analysis["Shipping Delay"] = (
    analysis["Actual Shipping Days"]
    - analysis["Days for shipment (scheduled)"]
)

analysis["Absolute Prediction Error"] = abs(
    analysis["Actual Shipping Days"]
    - analysis["Predicted Shipping Days"]
)

# Overall logistics performance
print("\n" + "=" * 65)
print("OVERALL LOGISTICS PERFORMANCE")
print("=" * 65)

total_orders = len(analysis)

delayed_orders = (
    analysis["Shipping Delay"] > 0
).sum()

on_or_before_schedule = (
    analysis["Shipping Delay"] <= 0
).sum()

delay_percentage = (
    delayed_orders / total_orders
) * 100

average_delay = analysis["Shipping Delay"].mean()

print("Test orders analysed:", total_orders)
print("Orders exceeding scheduled days:", delayed_orders)
print("Orders on or before schedule:", on_or_before_schedule)
print("Percentage exceeding schedule:", round(delay_percentage, 2), "%")
print("Average actual minus scheduled days:", round(average_delay, 3))

# Shipping mode analysis
shipping_analysis = analysis.groupby(
    "Shipping Mode"
).agg(
    Orders=("Actual Shipping Days", "count"),
    Average_Actual_Days=("Actual Shipping Days", "mean"),
    Average_Scheduled_Days=(
        "Days for shipment (scheduled)", "mean"
    ),
    Average_Delay=("Shipping Delay", "mean"),
    Average_Model_Error=("Absolute Prediction Error", "mean")
).round(3)

shipping_analysis = shipping_analysis.sort_values(
    by="Average_Delay",
    ascending=False
)

print("\n" + "=" * 65)
print("PERFORMANCE BY SHIPPING MODE")
print("=" * 65)

print(shipping_analysis.to_string())

# Market analysis
market_analysis = analysis.groupby(
    "Market"
).agg(
    Orders=("Actual Shipping Days", "count"),
    Average_Actual_Days=("Actual Shipping Days", "mean"),
    Average_Scheduled_Days=(
        "Days for shipment (scheduled)", "mean"
    ),
    Average_Delay=("Shipping Delay", "mean")
).round(3)

market_analysis = market_analysis.sort_values(
    by="Average_Delay",
    ascending=False
)

print("\n" + "=" * 65)
print("PERFORMANCE BY MARKET")
print("=" * 65)

print(market_analysis.to_string())

# Regional analysis: focus on regions with at least 100 test orders
region_analysis = analysis.groupby(
    "Order Region"
).agg(
    Orders=("Actual Shipping Days", "count"),
    Average_Actual_Days=("Actual Shipping Days", "mean"),
    Average_Scheduled_Days=(
        "Days for shipment (scheduled)", "mean"
    ),
    Average_Delay=("Shipping Delay", "mean")
)

region_analysis = region_analysis[
    region_analysis["Orders"] >= 100
].sort_values(
    by="Average_Delay",
    ascending=False
).round(3)

print("\n" + "=" * 65)
print("REGIONAL PERFORMANCE (MINIMUM 100 ORDERS)")
print("=" * 65)

print(region_analysis.to_string())

# Model performance by shipping mode
model_analysis = analysis.groupby(
    "Shipping Mode"
).agg(
    Orders=("Absolute Prediction Error", "count"),
    MAE=("Absolute Prediction Error", "mean")
).round(3)

print("\n" + "=" * 65)
print("MODEL PREDICTION ERROR BY SHIPPING MODE")
print("=" * 65)

print(model_analysis.to_string())

# Save analytical tables
shipping_analysis.to_csv(
    OUTPUT_DIR / "optimization_by_shipping_mode.csv"
)

market_analysis.to_csv(
    OUTPUT_DIR / "optimization_by_market.csv"
)

region_analysis.to_csv(
    OUTPUT_DIR / "optimization_by_region.csv"
)

model_analysis.to_csv(
    OUTPUT_DIR / "prediction_error_by_shipping_mode.csv"
)

# Chart 1: Average delay by shipping mode
plt.figure(figsize=(10, 6))

sns.barplot(
    data=shipping_analysis.reset_index(),
    x="Shipping Mode",
    y="Average_Delay"
)

plt.title("Average Shipping Delay by Shipping Mode")
plt.xlabel("Shipping Mode")
plt.ylabel("Actual Minus Scheduled Days")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "optimization_shipping_mode.png",
    dpi=300
)

plt.close()

# Chart 2: Average delay by market
plt.figure(figsize=(11, 6))

sns.barplot(
    data=market_analysis.reset_index(),
    x="Market",
    y="Average_Delay"
)

plt.title("Average Shipping Delay by Market")
plt.xlabel("Market")
plt.ylabel("Actual Minus Scheduled Days")
plt.xticks(rotation=25)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "optimization_market.png",
    dpi=300
)

plt.close()

# Chart 3: Actual versus scheduled shipping days
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=analysis.sample(
        min(3000, len(analysis)),
        random_state=42
    ),
    x="Days for shipment (scheduled)",
    y="Actual Shipping Days",
    alpha=0.4
)

plt.title("Actual versus Scheduled Shipping Days")
plt.xlabel("Scheduled Shipping Days")
plt.ylabel("Actual Shipping Days")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "actual_vs_scheduled_shipping.png",
    dpi=300
)

plt.close()

# Business recommendations based on measured results
print("\n" + "=" * 65)
print("DATA-BASED OPTIMIZATION RECOMMENDATIONS")
print("=" * 65)

if delay_percentage > 0:
    print(
        f"1. Review the causes of schedule overruns: "
        f"{delay_percentage:.2f}% of test orders "
        "exceeded their scheduled shipping days."
    )
else:
    print(
        "1. Most test orders met or beat their scheduled "
        "shipping duration. Continue monitoring performance."
    )

worst_mode = shipping_analysis.index[0]
worst_mode_delay = shipping_analysis.iloc[0]["Average_Delay"]

print(
    f"2. Review the {worst_mode} shipping process. "
    f"It has the highest average actual-minus-scheduled "
    f"duration ({worst_mode_delay:.2f} days) among the "
    "shipping modes in this test sample."
)

if len(market_analysis) > 0:
    worst_market = market_analysis.index[0]
    worst_market_delay = market_analysis.iloc[0]["Average_Delay"]

    print(
        f"3. Investigate logistics operations in {worst_market}. "
        f"It has the highest average actual-minus-scheduled "
        f"duration ({worst_market_delay:.2f} days) among the "
        "markets analysed."
    )

print(
    "4. Use the prediction model as a planning aid, "
    "not as a guaranteed delivery estimate."
)

print(
    "5. Collect additional operational information, "
    "such as warehouse processing time, carrier performance, "
    "transport distance and inventory availability, "
    "to improve future predictions."
)

print("\nAll optimization tables and charts saved successfully.")

print("\nSTEP 6 COMPLETED")

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 65)
print("PREDICTIVE LOGISTICS OPTIMIZATION")
print("LOGISTICS PERFORMANCE AND OPTIMIZATION ANALYSIS")
print("=" * 65)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "logistics_modeling_data.csv"
)

PREDICTION_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "validated_test_predictions.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
FIGURE_DIR = PROJECT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Load datasets
print("\nLoading prepared data and test predictions...")

df = pd.read_csv(DATA_PATH)
predictions = pd.read_csv(PREDICTION_PATH)

# Recreate the original test split to align predictions
from sklearn.model_selection import train_test_split

target = "Days for shipping (real)"

X = df.drop(columns=[target])
y = df[target]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Attach test features to predictions
analysis = X_test.reset_index(drop=True).copy()

analysis["Actual Shipping Days"] = predictions[
    "Actual Shipping Days"
].to_numpy()

analysis["Predicted Shipping Days"] = predictions[
    "Predicted Shipping Days"
].to_numpy()

# Calculate shipping duration difference
analysis["Shipping Delay"] = (
    analysis["Actual Shipping Days"]
    - analysis["Days for shipment (scheduled)"]
)

analysis["Absolute Prediction Error"] = abs(
    analysis["Actual Shipping Days"]
    - analysis["Predicted Shipping Days"]
)

# Overall logistics performance
print("\n" + "=" * 65)
print("OVERALL LOGISTICS PERFORMANCE")
print("=" * 65)

total_orders = len(analysis)

delayed_orders = (
    analysis["Shipping Delay"] > 0
).sum()

on_or_before_schedule = (
    analysis["Shipping Delay"] <= 0
).sum()

delay_percentage = (
    delayed_orders / total_orders
) * 100

average_delay = analysis["Shipping Delay"].mean()

print("Test orders analysed:", total_orders)
print("Orders exceeding scheduled days:", delayed_orders)
print("Orders on or before schedule:", on_or_before_schedule)
print("Percentage exceeding schedule:", round(delay_percentage, 2), "%")
print("Average actual minus scheduled days:", round(average_delay, 3))

# Shipping mode analysis
shipping_analysis = analysis.groupby(
    "Shipping Mode"
).agg(
    Orders=("Actual Shipping Days", "count"),
    Average_Actual_Days=("Actual Shipping Days", "mean"),
    Average_Scheduled_Days=(
        "Days for shipment (scheduled)", "mean"
    ),
    Average_Delay=("Shipping Delay", "mean"),
    Average_Model_Error=("Absolute Prediction Error", "mean")
).round(3)

shipping_analysis = shipping_analysis.sort_values(
    by="Average_Delay",
    ascending=False
)

print("\n" + "=" * 65)
print("PERFORMANCE BY SHIPPING MODE")
print("=" * 65)

print(shipping_analysis.to_string())

# Market analysis
market_analysis = analysis.groupby(
    "Market"
).agg(
    Orders=("Actual Shipping Days", "count"),
    Average_Actual_Days=("Actual Shipping Days", "mean"),
    Average_Scheduled_Days=(
        "Days for shipment (scheduled)", "mean"
    ),
    Average_Delay=("Shipping Delay", "mean")
).round(3)

market_analysis = market_analysis.sort_values(
    by="Average_Delay",
    ascending=False
)

print("\n" + "=" * 65)
print("PERFORMANCE BY MARKET")
print("=" * 65)

print(market_analysis.to_string())

# Regional analysis: focus on regions with at least 100 test orders
region_analysis = analysis.groupby(
    "Order Region"
).agg(
    Orders=("Actual Shipping Days", "count"),
    Average_Actual_Days=("Actual Shipping Days", "mean"),
    Average_Scheduled_Days=(
        "Days for shipment (scheduled)", "mean"
    ),
    Average_Delay=("Shipping Delay", "mean")
)

region_analysis = region_analysis[
    region_analysis["Orders"] >= 100
].sort_values(
    by="Average_Delay",
    ascending=False
).round(3)

print("\n" + "=" * 65)
print("REGIONAL PERFORMANCE (MINIMUM 100 ORDERS)")
print("=" * 65)

print(region_analysis.to_string())

# Model performance by shipping mode
model_analysis = analysis.groupby(
    "Shipping Mode"
).agg(
    Orders=("Absolute Prediction Error", "count"),
    MAE=("Absolute Prediction Error", "mean")
).round(3)

print("\n" + "=" * 65)
print("MODEL PREDICTION ERROR BY SHIPPING MODE")
print("=" * 65)

print(model_analysis.to_string())

# Save analytical tables
shipping_analysis.to_csv(
    OUTPUT_DIR / "optimization_by_shipping_mode.csv"
)

market_analysis.to_csv(
    OUTPUT_DIR / "optimization_by_market.csv"
)

region_analysis.to_csv(
    OUTPUT_DIR / "optimization_by_region.csv"
)

model_analysis.to_csv(
    OUTPUT_DIR / "prediction_error_by_shipping_mode.csv"
)

# Chart 1: Average delay by shipping mode
plt.figure(figsize=(10, 6))

sns.barplot(
    data=shipping_analysis.reset_index(),
    x="Shipping Mode",
    y="Average_Delay"
)

plt.title("Average Shipping Delay by Shipping Mode")
plt.xlabel("Shipping Mode")
plt.ylabel("Actual Minus Scheduled Days")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "optimization_shipping_mode.png",
    dpi=300
)

plt.close()

# Chart 2: Average delay by market
plt.figure(figsize=(11, 6))

sns.barplot(
    data=market_analysis.reset_index(),
    x="Market",
    y="Average_Delay"
)

plt.title("Average Shipping Delay by Market")
plt.xlabel("Market")
plt.ylabel("Actual Minus Scheduled Days")
plt.xticks(rotation=25)
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "optimization_market.png",
    dpi=300
)

plt.close()

# Chart 3: Actual versus scheduled shipping days
plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=analysis.sample(
        min(3000, len(analysis)),
        random_state=42
    ),
    x="Days for shipment (scheduled)",
    y="Actual Shipping Days",
    alpha=0.4
)

plt.title("Actual versus Scheduled Shipping Days")
plt.xlabel("Scheduled Shipping Days")
plt.ylabel("Actual Shipping Days")
plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "actual_vs_scheduled_shipping.png",
    dpi=300
)

plt.close()

# Business recommendations based on measured results
print("\n" + "=" * 65)
print("DATA-BASED OPTIMIZATION RECOMMENDATIONS")
print("=" * 65)

if delay_percentage > 0:
    print(
        f"1. Review the causes of schedule overruns: "
        f"{delay_percentage:.2f}% of test orders "
        "exceeded their scheduled shipping days."
    )
else:
    print(
        "1. Most test orders met or beat their scheduled "
        "shipping duration. Continue monitoring performance."
    )

worst_mode = shipping_analysis.index[0]
worst_mode_delay = shipping_analysis.iloc[0]["Average_Delay"]

print(
    f"2. Review the {worst_mode} shipping process. "
    f"It has the highest average actual-minus-scheduled "
    f"duration ({worst_mode_delay:.2f} days) among the "
    "shipping modes in this test sample."
)

if len(market_analysis) > 0:
    worst_market = market_analysis.index[0]
    worst_market_delay = market_analysis.iloc[0]["Average_Delay"]

    print(
        f"3. Investigate logistics operations in {worst_market}. "
        f"It has the highest average actual-minus-scheduled "
        f"duration ({worst_market_delay:.2f} days) among the "
        "markets analysed."
    )

print(
    "4. Use the prediction model as a planning aid, "
    "not as a guaranteed delivery estimate."
)

print(
    "5. Collect additional operational information, "
    "such as warehouse processing time, carrier performance, "
    "transport distance and inventory availability, "
    "to improve future predictions."
)

print("\nAll optimization tables and charts saved successfully.")

print("\nSTEP 6 COMPLETED")

