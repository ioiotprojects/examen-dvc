from pathlib import Path
import json
import pickle

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def main():
    ROOT = Path(__file__).resolve().parents[2]
    DATA = ROOT / "data"
    PROCESSED = DATA / "processed"
    MODELS = ROOT / "models"
    METRICS = ROOT / "metrics"

    X_test = pd.read_csv(PROCESSED / "X_test_scaled.csv")
    y_test = pd.read_csv(PROCESSED / "y_test.csv")["silica_concentrate"]

    with open(MODELS / "model.pkl", "rb") as file:
        model = pickle.load(file)

    y_pred = model.predict(X_test)

    predictions = pd.DataFrame({
        "y_true": y_test.to_numpy(),
        "y_pred": y_pred,
    })
    predictions.to_csv(DATA / "predictions.csv", index=False)

    scores = {
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "R2": float(r2_score(y_test, y_pred)),
    }

    METRICS.mkdir(parents=True, exist_ok=True)

    with open(METRICS / "scores.json", "w", encoding="utf-8") as file:
        json.dump(scores, file, indent=4, allow_nan=False)

    print("Evaluation scores:", scores)
    print("Predictions saved to:", DATA / "predictions.csv")
    print("Scores saved to:", METRICS / "scores.json")


if __name__ == "__main__":
    main()