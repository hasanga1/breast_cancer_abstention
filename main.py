import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from mapie.classification import SplitConformalClassifier

def load_and_prepare_data():
    # 1. Load the Wisconsin dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target # 0 = malignant, 1 = benign

    # 2. Split into Train, Calibration, and Test sets
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=0.3, random_state=42, stratify=y_temp
    )

    # 3. Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_calib_scaled = scaler.transform(X_calib)
    X_test_scaled = scaler.transform(X_test)

    print("Data preparation complete.")
    return X_train_scaled, X_calib_scaled, X_test_scaled, y_train, y_calib, y_test

def train_and_calibrate(X_train, y_train, X_calib, y_calib):
    # Step 2: Base Training [cite: 37]
    print("\n--- Training Base Model ---")
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    print("Base Random Forest trained successfully.")

    # Step 3: Calibration Phase [cite: 38]
    print("\n--- Calibrating with Conformal Prediction ---")
    # SplitConformalClassifier with prefit=True uses the pre-fitted model
    mapie_clf = SplitConformalClassifier(
        estimator=base_model, 
        prefit=True, 
        conformity_score='lac'
    )
    
    # Conformalize on the hold-out calibration dataset to calculate non-conformity scores 
    mapie_clf.conformalize(X_calib, y_calib)
    print("Conformal Prediction calibration complete.")
    
    return base_model, mapie_clf

if __name__ == "__main__":
    X_train, X_calib, X_test, y_train, y_calib, y_test = load_and_prepare_data()
    base_model, mapie_model = train_and_calibrate(X_train, y_train, X_calib, y_calib)