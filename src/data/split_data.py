from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "raw_data" / "raw.csv"
OUTPUT = ROOT / "data" / "processed"

if not INPUT.is_file():
    raise FileNotFoundError(f"Run collect.sh first. Missing file: {file.resolve()}")

df = pd.read_csv(INPUT)
print(df.head())

y = df['silica_concentrate']
X = df.drop(columns=['date', 'silica_concentrate'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

OUTPUT.mkdir(parents=True, exist_ok=True)

X_train.to_csv(OUTPUT / "X_train.csv", index=False)
X_test.to_csv(OUTPUT / "X_test.csv", index=False)
y_train.to_csv(OUTPUT / "y_train.csv", index=False)
y_test.to_csv(OUTPUT / "y_test.csv", index=False)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Saved datasets to: {OUTPUT}")


