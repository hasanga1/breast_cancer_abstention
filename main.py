import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data():
    # 1. Load the Wisconsin dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target # 0 = malignant, 1 = benign

    # 2. Split into Train, Calibration, and Test sets
    # First, hold out 20% for the final Test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Next, split the remaining 80% into Train (70%) and Calibration (30%)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=0.3, random_state=42, stratify=y_temp
    )

    # 3. Scale the features
    # Standardizing features is critical for MLPs and helpful for overall model stability
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_calib_scaled = scaler.transform(X_calib)
    X_test_scaled = scaler.transform(X_test)

    print("Data preparation complete.")
    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Calibration set: {X_calib_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")

    return X_train_scaled, X_calib_scaled, X_test_scaled, y_train, y_calib, y_test

if __name__ == "__main__":
    X_train, X_calib, X_test, y_train, y_calib, y_test = load_and_prepare_data()