import pandas as pd
import joblib
import json
import os
import sys

MODEL_DIR = r"C:\Users\Albostan\Projects\adnoc_lithology_ml\models"

def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "random_forest_lithology.joblib"))
    with open(os.path.join(MODEL_DIR, "train_medians.json")) as f:
        medians = json.load(f)
    with open(os.path.join(MODEL_DIR, "physical_ranges.json")) as f:
        ranges = json.load(f)
    with open(os.path.join(MODEL_DIR, "features.json")) as f:
        features = json.load(f)
    return model, medians, ranges, features

def predict_lithology(input_csv_path, output_csv_path):
    model, medians, ranges, features = load_artifacts()
    df = pd.read_csv(input_csv_path, low_memory=False)

    missing_cols = [f for f in features if f not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    X_new = df[features].copy()
    for log, (lo, hi) in ranges.items():
        X_new.loc[(X_new[log] < lo) | (X_new[log] > hi), log] = pd.NA
    for col in features:
        X_new[col] = X_new[col].fillna(medians[col])

    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new).max(axis=1)

    df["PREDICTED_LITHOLOGY"] = predictions
    df["PREDICTION_CONFIDENCE"] = probabilities.round(3)
    df.to_csv(output_csv_path, index=False)

    print(f"Predictions saved to: {output_csv_path}")
    print(f"Rows processed: {len(df)}")
    print("\nPrediction distribution:")
    print(df["PREDICTED_LITHOLOGY"].value_counts())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict.py <input_csv> <output_csv>")
        sys.exit(1)
    predict_lithology(sys.argv[1], sys.argv[2])