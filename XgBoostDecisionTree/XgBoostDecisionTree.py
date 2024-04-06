import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
import xgboost as xgb

def xgboost(data, biomarkers):

    # Get only the three columns that were used in the initial research
    X = data[biomarkers]
    y = data['outcome']

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Initialize XGBoost model
    xgb_model = xgb.XGBClassifier()

    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Cross-validation using StratifiedKFold
    xgb_scores = cross_val_score(xgb_model, X_imputed, y, cv=skf, scoring='accuracy')
    cv_precision = cross_val_score(xgb_model, X_imputed, y, cv=skf, scoring='precision')
    cv_recall = cross_val_score(xgb_model, X_imputed, y, cv=skf, scoring='recall')
    cv_f1_score = cross_val_score(xgb_model, X_imputed, y, cv=skf, scoring='f1')
    cv_auc = cross_val_score(xgb_model, X_imputed, y, cv=skf, scoring='roc_auc')

    # Calculate mean of additional evaluation metrics
    mean_accuracy = np.mean(xgb_scores)
    mean_precision = np.mean(cv_precision)
    mean_recall = np.mean(cv_recall)
    mean_f1_score = np.mean(cv_f1_score)
    mean_auc = np.mean(cv_auc)

    return mean_accuracy, mean_precision, mean_recall, mean_f1_score, mean_auc

