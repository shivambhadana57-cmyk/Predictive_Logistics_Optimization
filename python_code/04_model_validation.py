
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("=" * 65)
print("PREDICTIVE LOGISTICS OPTIMIZATION")
print("MODEL VALIDATION AND HYPERPARAMETER TUNING")
print("=" * 65)

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_DIR / "data" / "processed"
    / "logistics_modeling_data.csv"
)

MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load prepared data
print("\nLoading prepared dataset...")
df = pd.read_csv(DATA_PATH)

target = "Days for shipping (real)"

X = df.drop(columns=[target])
y = df[target]

# Identify feature types
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# Reproduce the original train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Preprocessing
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

# Linear Regression benchmark
linear_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

print("\nTraining Linear Regression benchmark...")
linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

# Ridge Regression pipeline
ridge_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge())
])

# Test different alpha values using cross-validation
parameter_grid = {
    "model__alpha": [0.1, 1.0, 10.0, 100.0]
}

print("\nStarting 3-fold cross-validation...")
print("Testing Ridge alpha values:", parameter_grid["model__alpha"])

grid_search = GridSearchCV(
    estimator=ridge_pipeline,
    param_grid=parameter_grid,
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_ridge = grid_search.best_estimator_

print("\nBest Ridge parameters:")
print(grid_search.best_params_)

print(
    "Best cross-validation RMSE:",
    round(-grid_search.best_score_, 4)
)

# Evaluate on the untouched test set
ridge_predictions = best_ridge.predict(X_test)

def evaluate_model(name, actual, predicted):
    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "R2 Score": r2_score(actual, predicted)
    }

results = pd.DataFrame([
    evaluate_model(
        "Linear Regression",
        y_test,
        linear_predictions
    ),
    evaluate_model(
        "Tuned Ridge Regression",
        y_test,
        ridge_predictions
    )
])

results = results.sort_values(
    by="RMSE",
    ascending=True
)

print("\n" + "=" * 65)
print("VALIDATION RESULTS")
print("=" * 65)

print(results.to_string(index=False))

# Select the better model based on test RMSE
if (
    results.iloc[0]["Model"]
    == "Tuned Ridge Regression"
):
    best_model = best_ridge
    best_predictions = ridge_predictions
    best_name = "Tuned Ridge Regression"
else:
    best_model = linear_model
    best_predictions = linear_predictions
    best_name = "Linear Regression"

print("\nSelected model:", best_name)

# Save validation results
results_path = OUTPUT_DIR / "model_validation_results.csv"
results.to_csv(results_path, index=False)

# Save selected model
model_path = MODEL_DIR / "final_shipping_days_model.joblib"
joblib.dump(best_model, model_path)

# Save test predictions
prediction_results = pd.DataFrame({
    "Actual Shipping Days": y_test.to_numpy(),
    "Predicted Shipping Days": best_predictions
})

prediction_path = OUTPUT_DIR / "validated_test_predictions.csv"
prediction_results.to_csv(prediction_path, index=False)

print("\nValidation results saved:", results_path)
print("Selected model saved:", model_path)
print("Test predictions saved:", prediction_path)

print("\nSTEP 5 COMPLETED")
