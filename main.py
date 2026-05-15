import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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

    return X_train_scaled, X_calib_scaled, X_test_scaled, y_train, y_calib, y_test

def train_and_calibrate(X_train, y_train, X_calib, y_calib, conf_level):
    # Step 2: Base Training
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)

    # Step 3: Calibration Phase using the specific confidence level
    mapie_clf = SplitConformalClassifier(
        estimator=base_model, 
        prefit=True, 
        conformity_score='lac',
        confidence_level=conf_level 
    )
    
    mapie_clf.conformalize(X_calib, y_calib)
    return base_model, mapie_clf

def evaluate_and_abstain(base_model, mapie_model, X_test, y_test, conf_level):
    print(f"\n--- Results for {conf_level*100:.1f}% Confidence ---")
    
    # 1. Get Base Model Predictions
    base_y_pred = base_model.predict(X_test)
    base_acc = accuracy_score(y_test, base_y_pred)
    
    # 2. Get Conformal Prediction Sets
    cp_y_pred, cp_y_ps = mapie_model.predict_set(X_test)
    
    # Slice the array to get a 2D boolean array: (n_samples, n_classes)
    y_ps = cp_y_ps[:, :, 0] if cp_y_ps.ndim == 3 else cp_y_ps
    
    # Calculate the size of each prediction set
    set_sizes = y_ps.sum(axis=1)
    
    # 3. Selective Abstention Logic
    accepted_mask = (set_sizes == 1)
    rejected_mask = (set_sizes > 1) | (set_sizes == 0)
    
    total_samples = len(y_test)
    accepted_count = np.sum(accepted_mask)
    rejected_count = np.sum(rejected_mask)
    
    # Calculate accuracy on accepted samples only
    if accepted_count > 0:
        accepted_acc = accuracy_score(y_test[accepted_mask], base_y_pred[accepted_mask])
    else:
        accepted_acc = 0.0

    # 4. Print the Report
    print(f"Base Model Accuracy: {base_acc*100:.2f}%")
    print(f"Samples Accepted: {accepted_count} ({(accepted_count/total_samples)*100:.2f}%)")
    print(f"Samples Rejected: {rejected_count} ({(rejected_count/total_samples)*100:.2f}%)")
    print(f"Accuracy on Accepted Samples: {accepted_acc*100:.2f}%")

if __name__ == "__main__":
    print("Preparing data...")
    X_train, X_calib, X_test, y_train, y_calib, y_test = load_and_prepare_data()
    
    # Test a realistic range of confidence levels
    confidence_levels = [0.80, 0.90, 0.95, 0.99]
    
    for conf in confidence_levels:
        base_model, mapie_model = train_and_calibrate(X_train, y_train, X_calib, y_calib, conf)
        evaluate_and_abstain(base_model, mapie_model, X_test, y_test, conf)