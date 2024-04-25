import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
from sklearn.impute import SimpleImputer

def decision_tree(data, biomarkers):
    
    # Get only columns needed from biomarker selection, if it is all, take all columns except the ones needed
    if biomarkers == "all":
        X = np.concatenate([data.iloc[:, [1, 2]].values, data.iloc[:, 6:]], axis=1)
    else:
        X = data[biomarkers]

    y = data['outcome']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Fit decision tree model
    model = DecisionTreeClassifier()

    # Perform cross-validation
    kf = KFold(n_splits=10, shuffle=True)
    y_pred = cross_val_predict(model, X_imputed, y, cv=kf)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Compute AUC score
    auc_scores = cross_val_score(model, X_imputed, y, cv=kf, scoring='roc_auc')
    mean_auc = np.mean(auc_scores)

    # Calculate mean cross-validation scores
    mean_acc = (tp + tn) / (tp + fp + tn + fn)
    mean_precision = tp / (tp + fp)
    mean_recall = tp / (tp + fn)
    mean_f1 = 2 * ((mean_precision * mean_recall) / (mean_precision + mean_recall))

    # Return accuracy, precision, recall, f1-score, mean AUC, and confusion matrix totals
    return mean_acc, mean_precision, mean_recall, mean_f1, mean_auc, tp, fp, tn, fn
