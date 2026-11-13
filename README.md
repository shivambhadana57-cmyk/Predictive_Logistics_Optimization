# Predictive Modeling and Optimization in Logistics Systems

## Project Overview

This project explores historical supply chain data to analyse shipping performance and develop a machine learning model for predicting actual shipping duration.

The project uses the DataCo SMART Supply Chain Dataset and applies Python-based data analytics techniques, including data preparation, exploratory data analysis, regression modelling, model validation, and logistics performance analysis.

## Objectives

- Understand and prepare a large logistics dataset.
- Explore shipping patterns across different shipping modes and geographical areas.
- Develop regression models to predict actual shipping duration.
- Evaluate model performance using MAE, RMSE, and R-squared.
- Compare actual shipping duration with scheduled shipping duration.
- Identify logistics performance patterns and provide business recommendations.

## Dataset

**Dataset:** DataCo SMART Supply Chain Dataset  
**Source:** Mendeley Data  
**Link:** https://data.mendeley.com/datasets/8gx2fvg2k6/5

The original dataset contains approximately 180,519 records and 53 columns.

The dataset is not included in this repository. Download it from the original source and place the required CSV file in the local `data/raw/` directory.

## Tools and Technologies

- Python 3.13.17
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Visual Studio Code

## Project Workflow

1. Dataset loading and inspection
2. Data preparation and feature selection
3. Exploratory Data Analysis (EDA)
4. Model training
5. Model validation and evaluation
6. Logistics performance and optimization analysis
7. Model interpretation and business recommendations

## Model Performance

The final selected model was Linear Regression.

| Evaluation Metric | Result |
|---|---:|
| MAE | 0.9921 days |
| RMSE | 1.2657 days |
| R-squared | 0.3923 |

The model was evaluated on a test dataset containing 36,104 records.

## Key Findings

- Average actual shipping duration was approximately 3.5 days.
- 57.35% of the test orders exceeded their scheduled shipping duration.
- Second Class had the highest average actual-minus-scheduled shipping duration, at approximately 1.973 days.
- Shipping performance differences were also observed across markets and geographical regions.

These findings describe patterns in the historical dataset and do not establish the causes of shipping delays.

## Repository Structure

- `data/` – Dataset location and data documentation.
- `python_code/` – Python scripts used for analysis and modelling.
- `figures/` – Generated charts and visualizations.
- `models/` – Saved machine learning model documentation and artifacts.
- `report/` – Internship project report and supporting screenshots.

## How to Run the Project

1. Clone or download this repository.
2. Download the dataset from the source link.
3. Place the required CSV file in `data/raw/`.
4. Create and activate a Python virtual environment.
5. Install the required libraries:

   `pip install -r requirements.txt`

6. Run the Python scripts in the project workflow order, following the instructions in `python_code/README.md`.

The scripts may require the dataset and generated intermediate files to be available in the expected locations.

## Limitations

The final model achieved moderate predictive performance. The analysis is based on historical data and does not include all operational factors, such as detailed warehouse processing times, carrier performance, transportation distance, or live traffic conditions.

The model should be treated as an analytical decision-support experiment, not a guaranteed shipping-time prediction system.

## Author

**Project Type:** Data Analytics Internship Project  
**Project Title:** Predictive Modeling and Optimization in Logistics Systems
