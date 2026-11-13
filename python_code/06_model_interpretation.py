
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 65)
print("MODEL INTERPRETATION AND PERFORMANCE ANALYSIS")
print("=" * 65)

# --------------------------------------------------
# 1. LOAD DATA AND FINAL MODEL
# --------------------------------------------------

data_path = "data/processed/logistics_modeling_data.csv"
model_path = "models/final_shipping_days_model.joblib"

df = pd.read_csv(data_path)

X = df.drop(columns=["Days for shipping (real)"])
y = df["Days for shipping (real)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = load(model_path)

print("\nDataset loaded:", df.shape)
print("Test records:", len(X_test))

# --------------------------------------------------
# 2. MODEL PREDICTIONS
# --------------------------------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("-" * 40)
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# --------------------------------------------------
# 3. SAVE PREDICTIONS
# --------------------------------------------------

prediction_df = pd.DataFrame({
    "Actual_Shipping_Days": y_test.values,
    "Predicted_Shipping_Days": predictions,
    "Prediction_Error": y_test.values - predictions
})

os.makedirs("data/processed", exist_ok=True)
os.makedirs("figures", exist_ok=True)

prediction_df.to_csv(
    "data/processed/model_interpretation_predictions.csv",
    index=False
)

# --------------------------------------------------
# 4. ACTUAL VS PREDICTED CHART
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.25
)

minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Shipping Days")
plt.ylabel("Predicted Shipping Days")
plt.title("Actual vs Predicted Shipping Duration")
plt.tight_layout()

plt.savefig("figures/06_actual_vs_predicted.png", dpi=300)
plt.close()

# --------------------------------------------------
# 5. PREDICTION ERROR DISTRIBUTION
# --------------------------------------------------

errors = y_test.values - predictions

plt.figure(figsize=(8, 6))

sns.histplot(errors, bins=30, kde=True)

plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Number of Orders")
plt.title("Distribution of Model Prediction Errors")
plt.tight_layout()

plt.savefig("figures/07_prediction_error_distribution.png", dpi=300)
plt.close()


# --------------------------------------------------
# 6. LINEAR REGRESSION COEFFICIENT ANALYSIS
# --------------------------------------------------

# Get coefficients from the Linear Regression step
coefficients = model.named_steps["model"].coef_

# Get feature names after preprocessing and encoding
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute_Coefficient": np.abs(coefficients)
})

importance_df = importance_df.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

importance_df.to_csv(
    "data/processed/linear_regression_coefficients.csv",
    index=False
)

print("\nFEATURE COEFFICIENT ANALYSIS")
print("-" * 50)
print(importance_df.head(20).to_string(index=False))

# --------------------------------------------------
# 7. FEATURE COEFFICIENT CHART
# --------------------------------------------------

plot_df = importance_df.head(20).sort_values(
    by="Absolute_Coefficient",
    ascending=True
)

plt.figure(figsize=(10, 8))

plt.barh(
    plot_df["Feature"],
    plot_df["Coefficient"]
)

plt.axvline(x=0, linestyle="--")

plt.xlabel("Linear Regression Coefficient")
plt.ylabel("Encoded Feature")
plt.title("Top 20 Linear Regression Coefficients")
plt.tight_layout()

plt.savefig(
    "figures/08_linear_regression_coefficients.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 7. FEATURE COEFFICIENT CHART
# --------------------------------------------------

plot_df = importance_df.sort_values(
    by="Absolute_Coefficient",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    plot_df["Feature"],
    plot_df["Coefficient"]
)

plt.axvline(x=0, linestyle="--")

plt.xlabel("Linear Regression Coefficient")
plt.ylabel("Feature")
plt.title("Feature Coefficients - Linear Regression")
plt.tight_layout()

plt.savefig(
    "figures/08_linear_regression_coefficients.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 8. FINAL SUMMARY
# --------------------------------------------------

print("\nINTERPRETATION SUMMARY")
print("-" * 50)
print(f"Mean Absolute Error: {mae:.3f} days")
print(f"Root Mean Squared Error: {rmse:.3f} days")
print(f"R-squared: {r2:.3f}")

print("\nFiles saved successfully:")
print("- model_interpretation_predictions.csv")
print("- linear_regression_coefficients.csv")
print("- 06_actual_vs_predicted.png")
print("- 07_prediction_error_distribution.png")
print("- 08_linear_regression_coefficients.png")

print("\nSTEP 7 COMPLETED SUCCESSFULLY!")
