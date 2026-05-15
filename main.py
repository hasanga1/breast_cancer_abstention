import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mapie.classification import SplitConformalClassifier

def load_and_prepare_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target 

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=0.3, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_calib_scaled = scaler.transform(X_calib)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_calib_scaled, X_test_scaled, y_train, y_calib, y_test

def train_and_calibrate(X_train, y_train, X_calib, y_calib, conf_level):
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)

    mapie_clf = SplitConformalClassifier(
        estimator=base_model, 
        prefit=True, 
        conformity_score='lac',
        confidence_level=conf_level 
    )
    
    mapie_clf.conformalize(X_calib, y_calib)
    return base_model, mapie_clf

def evaluate_metrics(base_model, mapie_model, X_test, y_test):
    base_y_pred = base_model.predict(X_test)
    
    cp_y_pred, cp_y_ps = mapie_model.predict_set(X_test)
    y_ps = cp_y_ps[:, :, 0] if cp_y_ps.ndim == 3 else cp_y_ps
    
    set_sizes = y_ps.sum(axis=1)
    
    accepted_mask = (set_sizes == 1)
    rejected_mask = (set_sizes > 1) | (set_sizes == 0)
    
    total_samples = len(y_test)
    accepted_count = np.sum(accepted_mask)
    rejected_count = np.sum(rejected_mask)
    
    rejection_rate = rejected_count / total_samples
    
    if accepted_count > 0:
        accepted_acc = accuracy_score(y_test[accepted_mask], base_y_pred[accepted_mask])
    else:
        accepted_acc = np.nan # Use NaN so it breaks the line graph cleanly if 0 samples are accepted

    return accepted_acc, rejection_rate

def generate_plot():
    print("Generating trade-off visualization...")
    X_train, X_calib, X_test, y_train, y_calib, y_test = load_and_prepare_data()
    
    # Generate points from 50% to 99% confidence
    confidence_levels = np.linspace(0.50, 0.99, 50)
    
    accuracies = []
    rejection_rates = []
    
    # Get base accuracy for the baseline reference
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    base_acc = accuracy_score(y_test, base_model.predict(X_test))

    for conf in confidence_levels:
        _, mapie_model = train_and_calibrate(X_train, y_train, X_calib, y_calib, conf)
        acc, rej = evaluate_metrics(base_model, mapie_model, X_test, y_test)
        accuracies.append(acc)
        rejection_rates.append(rej)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(confidence_levels * 100, np.array(accuracies) * 100, label='Accuracy on Accepted Samples', color='green', linewidth=2)
    plt.plot(confidence_levels * 100, np.array(rejection_rates) * 100, label='Rejection Rate (Referred to Doctor)', color='red', linestyle='--', linewidth=2)
    plt.axhline(y=base_acc * 100, color='gray', linestyle=':', label=f'Base Model Accuracy ({base_acc*100:.1f}%)')

    plt.title('Selective Abstention in Breast Cancer Diagnosis', fontsize=14)
    plt.xlabel('Target Confidence Level (%)', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.legend(loc='center left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('abstention_plot.png')
    print("Plot saved successfully as 'abstention_plot.png'!")

if __name__ == "__main__":
    generate_plot()