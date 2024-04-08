import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer
import xgboost as xgb

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

    # Initialize StratifiedKFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validation using StratifiedKFold
    cv_acc = cross_val_score(xgb_model, X_imputed, y, cv=kf, scoring='accuracy')
    cv_precision = cross_val_score(xgb_model, X_imputed, y, cv=kf, scoring='precision')
    cv_recall = cross_val_score(xgb_model, X_imputed, y, cv=kf, scoring='recall')
    cv_f1_score = cross_val_score(xgb_model, X_imputed, y, cv=kf, scoring='f1')
    cv_auc = cross_val_score(xgb_model, X_imputed, y, cv=kf, scoring='roc_auc')

    # Calculate mean of additional evaluation metrics
    mean_accuracy = np.mean(cv_acc)
    mean_precision = np.mean(cv_precision)
    mean_recall = np.mean(cv_recall)
    mean_f1_score = np.mean(cv_f1_score)
    mean_auc = np.mean(cv_auc)

    # Return accuracy, precision, recall, f1-score, and AUC
    return mean_accuracy, mean_precision, mean_recall, mean_f1_score, mean_auc

