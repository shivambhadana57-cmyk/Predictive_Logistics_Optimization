
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("=" * 65)
print("PREDICTIVE LOGISTICS OPTIMIZATION")
print("MACHINE LEARNING MODEL TRAINING")
print("=" * 65)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "logistics_modeling_data.csv"
)

MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load prepared dataset
print("\nLoading prepared dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# Define target and predictors
target = "Days for shipping (real)"

X = df.drop(columns=[target])
y = df[target]

# Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

print("\nNumerical features:", numeric_features)
print("\nCategorical features:", categorical_features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=10
    ))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# Define models
models = {
    "Dummy Regressor": DummyRegressor(strategy="mean"),

    "Ridge Regression": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=1.0))
    ]),

    "Linear Regression": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])
}


# Train and evaluate models
results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "-" * 65)
    print("Training:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))

# Compare model results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(results_df.to_string(index=False))

# Select the best model using RMSE
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("\nBest model:", best_model_name)

# Save evaluation results
results_path = OUTPUT_DIR / "model_evaluation_results.csv"
results_df.to_csv(results_path, index=False)

# Save best model
model_path = MODEL_DIR / "best_shipping_days_model.joblib"
joblib.dump(best_model, model_path)

# Save predictions from best model
best_predictions = best_model.predict(X_test)

prediction_results = pd.DataFrame({
    "Actual Shipping Days": y_test.to_numpy(),
    "Predicted Shipping Days": best_predictions
})

prediction_path = OUTPUT_DIR / "test_predictions.csv"
prediction_results.to_csv(prediction_path, index=False)

print("\nSaved model evaluation:", results_path)
print("Saved best model:", model_path)
print("Saved test predictions:", prediction_path)

print("\nSTEP 4 COMPLETED")


