# train.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def load_data():
    """
    Default uses sklearn iris dataset so you can run without CSV.
    If you have a CSV at data/train.csv with a 'target' column, the CSV branch will run.
    """
    csv_path = os.path.join("data", "train.csv")
    if os.path.exists(csv_path):
        print(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        if "target" not in df.columns:
            raise ValueError("CSV found but no 'target' column. Rename the label column to 'target'.")
        X = df.drop(columns=["target"]).values
        y = df["target"].values
        feature_names = list(df.drop(columns=["target"]).columns)
        return X, y, feature_names

    # fallback: use iris dataset (works out of the box)
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    print("No data/train.csv found — using sklearn iris dataset for demo.")
    return X, y, feature_names

def train_and_save():
    X, y, feature_names = load_data()
    print("Data shapes:", X.shape, y.shape)
    unique, counts = np.unique(y, return_counts=True)
    print("Label distribution:", dict(zip(unique.tolist(), counts.tolist())))

    # split with stratify to keep class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # model - RandomForest is robust for small demos
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy on test set: {acc:.4f}")
    print("\nClassification report:\n", classification_report(y_test, preds, zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Save model + scaler as a dictionary for easy loading
    os.makedirs("models", exist_ok=True)
    artifact = {"model": model, "scaler": scaler, "feature_names": feature_names}
    joblib.dump(artifact, os.path.join("models", "model.pkl"))
    print("Saved model artifact to models/model.pkl")

if __name__ == "__main__":
    train_and_save()
