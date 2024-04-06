import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import KNNImputer, SimpleImputer

# Set the print options to suppress scientific notation
np.set_printoptions(suppress=True)

# Read the CSV file
df = pd.read_csv("data/time_series_375_preprocess_en.csv", delimiter=';')

# Replace all empty values with NaN
df = df.replace('None', np.NaN)

# Group by id and take the latest non NaN value
df = df.groupby('id').last()

# Convert string columns to float starting from the 7th column
for col in df.columns[6:]:
    if df[col].dtype == 'object':
        df[col] = df[col].str.replace(',', '.').astype(float)

# Impute missing values using KNNImputer for independent variables
imputer = SimpleImputer(strategy='mean')
x_independent_imputed = imputer.fit_transform(df.iloc[:, 6:])

x_independent = np.concatenate([df.iloc[:, [1, 2]].values, x_independent_imputed], axis=1)
y_dependent = df['outcome']

le = LabelEncoder()
x_independent[:, 0] = le.fit_transform(x_independent[:, 0])

# Feature Scaling
sc = StandardScaler()
x_independent_scaled = sc.fit_transform(x_independent)

# Initialize KNN classifier
classifier = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=41)

# Perform k-fold cross-validation manually
cv_scores = cross_val_score(classifier, x_independent_scaled, y_dependent, cv=kf)

# Print cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", np.mean(cv_scores))