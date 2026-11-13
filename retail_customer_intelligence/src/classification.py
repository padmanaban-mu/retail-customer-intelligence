import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier

def train_all_models(features_df, target_series):
    """
    Trains Decision Tree, Naive Bayes, and SVM models on the customer features.
    Computes performance metrics and a majority-class baseline.
    Returns the trained models, scaler, and a comprehensive metrics dictionary.
    """
    # Prepare X and y
    X = features_df.copy()
    y = target_series['TargetClass'].copy()
    feature_names = X.columns.tolist()
    
    # 1. Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 2. Fit Scaler ONLY on training data to prevent leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Dictionary to hold all trained models
    models = {}
    metrics = {}
    
    from sklearn.model_selection import cross_validate, StratifiedKFold
    
    # Helper function to evaluate and store metrics
    def evaluate_model(name, model, is_scaled=False):
        X_train_to_use = X_train_scaled if is_scaled else X_train
        X_test_to_use = X_test_scaled if is_scaled else X_test
        
        # 5-fold CV on training data
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = cross_validate(model, X_train_to_use, y_train, cv=cv, scoring=['accuracy', 'f1_macro'])
        cv_acc_mean = cv_results['test_accuracy'].mean()
        cv_acc_std = cv_results['test_accuracy'].std()
        cv_f1_mean = cv_results['test_f1_macro'].mean()
        cv_f1_std = cv_results['test_f1_macro'].std()
        
        model.fit(X_train_to_use, y_train)
        y_pred = model.predict(X_test_to_use)
        
        # Train vs Test Accuracy
        y_train_pred = model.predict(X_train_to_use)
        train_acc = accuracy_score(y_train, y_train_pred)
        
        metrics[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'train_accuracy': train_acc,
            'precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='macro', zero_division=0),
            'cv_acc_mean': cv_acc_mean,
            'cv_acc_std': cv_acc_std,
            'cv_f1_mean': cv_f1_mean,
            'cv_f1_std': cv_f1_std,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }
        models[name] = model

    # --- Train Baseline ---
    # Predicts the most frequent class every time
    evaluate_model('Baseline (Majority Class)', DummyClassifier(strategy='most_frequent'))
    
    # --- Train Decision Tree ---
    # Limiting depth to 3 to prevent overfitting and keep the plot readable
    dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42)
    evaluate_model('Decision Tree', dt_model, is_scaled=True) # Scaling isn't strictly required for DT, but good for consistency
    
    # Capture feature importances
    metrics['Decision Tree']['feature_importances'] = dict(zip(feature_names, dt_model.feature_importances_))
    
    # Generate tree plot figure
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(dt_model, feature_names=feature_names, class_names=dt_model.classes_, filled=True, ax=ax, fontsize=10)
    plt.title("Decision Tree Visualization (Max Depth=3)")
    plt.tight_layout()
    metrics['Decision Tree']['tree_plot'] = fig
    
    # --- Train Naive Bayes ---
    evaluate_model('Naive Bayes', GaussianNB(), is_scaled=True)
    
    # --- Train SVM ---
    # Using RBF kernel. Enabling probability=True (Platt scaling) so we can show confidence scores in the UI.
    # Note for Viva: Platt scaling requires internal cross-validation and is computationally expensive on large datasets, 
    # but manageable for ~3000 customers.
    base_svc = SVC(kernel='rbf', random_state=42)
    calibrated_svc = CalibratedClassifierCV(base_svc, ensemble=False)
    evaluate_model('SVM (RBF)', calibrated_svc, is_scaled=True)
    
    metrics['data_stats'] = {
        'train_counts': y_train.value_counts().to_dict(),
        'test_counts': y_test.value_counts().to_dict()
    }
    
    return models, scaler, metrics, feature_names

def predict_customer(models, scaler, model_name, feature_dict, feature_names):
    """
    Takes a dictionary of a single customer's features, applies the exact same
    scaling used during training, and returns the predicted class and probabilities.
    """
    # Convert input dict to a 1-row DataFrame in the correct column order
    df_input = pd.DataFrame([feature_dict], columns=feature_names)
    
    # Apply the pre-fitted scaler
    X_input_scaled = scaler.transform(df_input)
    
    model = models[model_name]
    
    # Predict
    prediction = model.predict(X_input_scaled)[0]
    
    # Get probabilities if the model supports it
    probabilities = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_input_scaled)[0]
        # Map probabilities to class names
        probabilities = dict(zip(model.classes_, probs))
        
    return prediction, probabilities
