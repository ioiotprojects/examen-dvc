from pathlib import Path
import pickle

import pandas as pd
from sklearn.svm import SVR


def main():
    ROOT = Path(__file__).resolve().parents[2]
    DATA = ROOT / "data" / "processed"
    MODELS = ROOT / "models"

    X_train = pd.read_csv(DATA / "X_train_scaled.csv")
    y_train = pd.read_csv(DATA / "y_train.csv")["silica_concentrate"]

    with open(MODELS / "best_params.pkl", "rb") as file:
        best_params = pickle.load(file)

    model = SVR(**best_params)
    model.fit(X_train, y_train)

    MODELS.mkdir(parents=True, exist_ok=True)

    with open(MODELS / "model.pkl", "wb") as file:
        pickle.dump(model, file)

    print("Training completed.")
    print("Parameters:", best_params)
    print("Model saved to:", MODELS / "model.pkl")


if __name__ == "__main__":
    main()