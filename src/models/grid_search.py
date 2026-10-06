from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def main():
    ROOT = Path(__file__).resolve().parents[2]
    DATA = ROOT / "data" / "processed"
    MODELS = ROOT / "models"
    
    X_train = pd.read_csv(DATA / "X_train.csv")
    y_train = pd.read_csv(DATA / "y_train.csv")["silica_concentrate"]
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR())
    ])
    
    param_grid = {
        "model__kernel": ["rbf"],
        "model__C": [1, 10, 100],
        "model__epsilon": [0.01, 0.1, 0.5],
        "model__gamma": ["scale", 0.01, 0.1],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=False,
        error_score="raise",
    )
    
    search.fit(X_train, y_train)
    
    # Save parameters directly usable with SVR(**best_params)
    best_params = {
        name.removeprefix("model__"): value
        for name, value in search.best_params_.items()
    }
    
    MODELS.mkdir(parents=True, exist_ok=True)

    with open(MODELS / "best_params.pkl", "wb") as file:
        pickle.dump(best_params, file)

    print("Best parameters:", best_params)
    print("Best cross-validation MSE:", -search.best_score_)
    print("Parameters saved to:", MODELS / "best_params.pkl")

    
if __name__ == "__main__":
    main()