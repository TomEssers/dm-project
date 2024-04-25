import numpy as np
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import confusion_matrix, roc_auc_score

def xgboost(data, biomarkers):

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

    # Initialize XGBoost model
    xgb_model = xgb.XGBClassifier()

    # Initialize KFold
    kf = KFold(n_splits=10, shuffle=True)

    # Perform k-fold cross-validation
    y_pred = cross_val_predict(xgb_model, X_imputed, y, cv=kf)

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Compute AUC score
    auc_scores = cross_val_score(xgb_model, X_imputed, y, cv=kf, scoring='roc_auc')
    mean_auc = np.mean(auc_scores)

    # Calculate mean cross-validation scores
    mean_accuracy = (tp + tn) / (tp + fp + tn + fn)
    mean_precision = tp / (tp + fp)
    mean_recall = tp / (tp + fn)
    mean_f1_score = 2 * ((mean_precision * mean_recall) / (mean_precision + mean_recall))

    # Return accuracy, precision, recall, f1-score, mean AUC, and confusion matrix totals
    return mean_accuracy, mean_precision, mean_recall, mean_f1_score, mean_auc, tp, fp, tn, fn