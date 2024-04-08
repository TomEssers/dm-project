from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np


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

    # Initialize StratifiedKFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Build the logistic regression model
    model = LogisticRegression()

    # Perform cross-validation
    cv_scores_acc = cross_val_score(model, X_scaled, y, cv=kf, scoring='accuracy')
    cv_scores_precision = cross_val_score(model, X_scaled, y, cv=kf, scoring='precision')
    cv_scores_recall = cross_val_score(model, X_scaled, y, cv=kf, scoring='recall')
    cv_scores_f1 = cross_val_score(model, X_scaled, y, cv=kf, scoring='f1')
    cv_scores_auc = cross_val_score(model, X_scaled, y, cv=kf, scoring='roc_auc')

    # Calculate and print mean cross-validation scores
    mean_acc = np.mean(cv_scores_acc)
    mean_precision = np.mean(cv_scores_precision)
    mean_recall = np.mean(cv_scores_recall)
    mean_f1 = np.mean(cv_scores_f1)
    mean_auc = np.mean(cv_scores_auc)

    # Return accuracy, precision, recall, f1-score, and AUC
    return mean_acc, mean_precision, mean_recall, mean_f1, mean_auc