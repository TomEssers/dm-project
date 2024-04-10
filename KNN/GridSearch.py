import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def knn(data, biomarkers, name):
    
    kfold_amount = 20
    name=name

    # Get only columns needed from biomarker selection, if it is all, take all columns except the ones needed
    if biomarkers == "all":
        X = np.concatenate([data.iloc[:, [1, 2]].values, data.iloc[:, 6:]], axis=1)
    else:
        X = data[biomarkers]

    y = data['outcome']

    # Reset index of y to match X
    y = y.reset_index(drop=True)

    # Pipeline for imputation, scaling, and KNN classification
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    # Define parameter grid for GridSearchCV
    param_grid = {
        'knn__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],  # Example values for k
        'knn__weights': ['uniform', 'distance'],  # Example values for weight function
        # Add other hyperparameters you want to tune
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=kfold_amount, scoring=['precision', 'recall'], refit='precision', verbose=1)

    # Fit GridSearchCV
    grid_search.fit(X, y)

    # Get best parameters and best estimator
    best_params = grid_search.best_params_
    best_estimator = grid_search.best_estimator_

    # Extract grid search results
    results = grid_search.cv_results_
    mean_precision = results['mean_test_precision']
    mean_recall = results['mean_test_recall']
    params = results['params']

    # Plot mean precision and recall for each parameter combination
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(params)), mean_precision, label='Mean Precision', marker='o')
    plt.plot(range(len(params)), mean_recall, label='Mean Recall', marker='o')
    plt.xlabel('K Nearest Neighbours')
    plt.ylabel('Mean Score')
    plt.title('KNN Grid Search Results (kfold = ' +  str(kfold_amount) + '), (biomarkers = ' + name + ')')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Return best parameters and estimator
    return best_params, best_estimator