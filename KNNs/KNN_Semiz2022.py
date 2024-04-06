import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer

# Read the CSV file
data = pd.read_csv("data/time_series_375_preprocess_en.csv", delimiter=';')

# Replace all empty values with NaN
data = data.replace('None', np.NaN)

# Group by id and take the latest non NaN value
data = data.groupby('id').last()

# Convert string columns to float starting from the 7th column
for col in data.columns[6:]:
    if data[col].dtype == 'object':
        data[col] = data[col].str.replace(',', '.').astype(float)

# Add the NLR column (dividing the number of neutrophils by the number of lymphocytes)
data['NLR'] = data['neutrophils count'] / data['lymphocyte count']

# Get all columns from the research
# Missing cardiac and lung values
# ALT = glutamic-pyruvic transaminase
X = data[['Hypersensitive c-reactive protein', 'procalcitonin', 'Interleukin 6',
        'lymphocyte count', 'neutrophils count', 'D-D dimer', 'ferritin', 'Red blood cell distribution width ',
        'aspartate aminotransferase', 'glutamic-pyruvic transaminase', 'Total bilirubin', 'albumin', 'NLR']]
y = data['outcome']

# Reset index of y to match X
y = y.reset_index(drop=True)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Initialize KNN classifier
classifier = KNeighborsClassifier(n_neighbors=21, metric='minkowski', p=2)

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=10)

# Perform k-fold cross-validation manually
cv_scores = cross_val_score(classifier, X_scaled, y, cv=kf)

# Print cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", np.mean(cv_scores))