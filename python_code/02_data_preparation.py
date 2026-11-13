
from pathlib import Path
import pandas as pd

print("=" * 65)
print("PREDICTIVE LOGISTICS OPTIMIZATION")
print("DATA PREPARATION AND FEATURE SELECTION")
print("=" * 65)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "logistics_modeling_data.csv"

# Load original dataset
print("\nLoading original dataset...")
df = pd.read_csv(
    RAW_PATH,
    sep=",",
    encoding="latin-1",
    low_memory=False
)

print("Original shape:", df.shape)

# Prediction target
target = "Days for shipping (real)"

# Features available around order placement
selected_features = [
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Product Price",
    "Order Item Discount Rate",
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Country",
    "Category Name",
    "Department Name",
    "Customer Segment",
    "Type",
    "order date (DateOrders)"
]

# Keep only selected features and target
data = df[selected_features + [target]].copy()

print("\nSelected columns:", len(data.columns))

# Convert order date into datetime
data["order date (DateOrders)"] = pd.to_datetime(
    data["order date (DateOrders)"],
    format="mixed",
    errors="coerce"
)

# Create useful date-based features
data["Order Month"] = data["order date (DateOrders)"].dt.month
data["Order Day Of Week"] = data["order date (DateOrders)"].dt.dayofweek
data["Order Hour"] = data["order date (DateOrders)"].dt.hour

# Remove the original date column after feature extraction
data.drop(columns=["order date (DateOrders)"], inplace=True)

# Remove rows with missing target values
before = len(data)
data.dropna(subset=[target], inplace=True)
removed = before - len(data)

# Ensure target is numeric
data[target] = pd.to_numeric(data[target], errors="coerce")
data.dropna(subset=[target], inplace=True)

# Display preparation results
print("\n" + "=" * 65)
print("DATA PREPARATION RESULTS")
print("=" * 65)

print("\nPrepared dataset shape:", data.shape)
print("Rows removed due to missing target:", removed)

print("\nMissing values in selected columns:")
print(data.isnull().sum())

print("\nTarget variable summary:")
print(data[target].describe())

print("\nTarget value distribution:")
print(data[target].value_counts().sort_index())

print("\nCategorical feature counts:")
categorical_columns = data.select_dtypes(
    include=["object", "str"]
).columns

for column in categorical_columns:
    print(f"{column}: {data[column].nunique()} unique values")

# Save prepared dataset
data.to_csv(OUTPUT_PATH, index=False)

print("\nPrepared dataset saved successfully!")
print("Output file:", OUTPUT_PATH)

print("\nFinal columns:")
for column in data.columns:
    print("-", column)

print("\nSTEP 2 COMPLETED")
