from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed"

X_train = pd.read_csv(INPUT/"X_train.csv")
X_test = pd.read_csv(INPUT/"X_test.csv")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)


X_train_scaled.to_csv(INPUT / "X_train_scaled.csv", index=False)
X_test_scaled.to_csv(INPUT / "X_test_scaled.csv", index=False)

print("Scaled training and test datasets saved.")
