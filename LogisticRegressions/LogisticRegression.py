import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def logistic_regression(data, biomarkers):

    # Get only columns needed from biomarker selection, if it is all, take all columns except the ones needed
    if biomarkers == "all":
        X = np.concatenate([data.iloc[:, [1, 2]].values, data.iloc[:, 6:]], axis=1)
    else:
        X = data[biomarkers]

    y = data['outcome']

    # Reset index of y to match X
    y = y.reset_index(drop=True)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Initialize logistic regression model
    model = LogisticRegression()

    # Initialize KFold
    kf = KFold(n_splits=10, shuffle=True)

    # Perform k-fold cross-validation
    y_pred = cross_val_predict(model, X_scaled, y, cv=kf)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Compute AUC score
    auc_scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='roc_auc')
    mean_auc = np.mean(auc_scores)

    # Calculate mean cross-validation scores
    mean_acc = (tp + tn) / (tp + fp + tn + fn)
    mean_precision = tp / (tp + fp)
    mean_recall = tp / (tp + fn)
    mean_f1 = 2 * ((mean_precision * mean_recall) / (mean_precision + mean_recall))

    # Return accuracy, precision, recall, f1-score, mean AUC, and confusion matrix totals
    return mean_acc, mean_precision, mean_recall, mean_f1, mean_auc, tp, fp, tn, fn