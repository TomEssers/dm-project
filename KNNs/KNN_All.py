import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.impute import SimpleImputer

# TODO Data science: 
# Graph voor knn proberen

# Nadelen van hoge k's knn
# kijken of randomized werkt knn
# KNN avgs nemen
# Misschien is de mean accuracy van de matrix niet het beste om te gebruiken als meetwaarde, 
# aangezien het om doden enzo gaan is bijvoorbeeld 
# de False Negative is erger dan de False Positive.

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

# Feature Scaling
sc = StandardScaler()
x_independent_scaled = sc.fit_transform(x_independent)

# Initialize KNN classifier
classifier = KNeighborsClassifier(n_neighbors=7, metric='minkowski', p=2)

# Initialize KFold with a shuffle parameter so the folds are random every run
kf = KFold(n_splits=5, shuffle=True, random_state=40)

# Perform k-fold cross-validation manually
cv_scores = cross_val_score(classifier, x_independent_scaled, y_dependent, cv=kf)

# Print cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", np.mean(cv_scores))