
# ============================================================
# PROJECT: Predictive Modeling and Optimization in Logistics
# FILE: 01_data_loading.py
# PURPOSE: Load and inspect the DataCo supply chain dataset
# ============================================================

import pandas as pd
from pathlib import Path

# 1. Define project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"

# 2. Check whether the dataset exists
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

print("=" * 65)
print("PREDICTIVE LOGISTICS OPTIMIZATION")
print("DATA LOADING AND INITIAL INSPECTION")
print("=" * 65)

# 3. Load the main dataset
print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep=",",
    encoding="latin-1",
    low_memory=False
)

# 4. Display dataset dimensions
print("\n1. DATASET DIMENSIONS")
print("-" * 40)
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

# 5. Display first 10 records
print("\n2. FIRST 10 RECORDS")
print("-" * 40)
print(df.head(10).to_string())

# 6. Display column names
print("\n3. COLUMN NAMES")
print("-" * 40)

for number, column in enumerate(df.columns, start=1):
    print(f"{number}. {column}")

# 7. Display data types
print("\n4. DATA TYPES")
print("-" * 40)
print(df.dtypes.to_string())

# 8. Check missing values
print("\n5. MISSING VALUES")
print("-" * 40)

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("No missing values found.")
else:
    print(missing.to_string())

# 9. Check duplicate records
print("\n6. DUPLICATE RECORDS")
print("-" * 40)
print("Duplicate Rows:", df.duplicated().sum())

# 10. Check numerical statistics
print("\n7. NUMERICAL SUMMARY")
print("-" * 40)
print(df.describe().to_string())

# 11. Check categorical information
print("\n8. DATASET INFORMATION")
print("-" * 40)
df.info()

# 12. Check potential prediction targets

print("\n9. POTENTIAL PREDICTION TARGETS")
print("-" * 40)

targets = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Late_delivery_risk",
    "Delivery Status"
]

for target in targets:
    if target in df.columns:
        print(f"\nTarget found: {target}")
        print(df[target].describe())
    else:
        print(f"\nNot found: {target}")