import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
from sklearn.impute import SimpleImputer

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

# Add the NLR column (dividing the number of neutrophils by the number of lymphocytes)
df['NLR'] = df['neutrophils count'] / df['lymphocyte count']

# Get only the biomarkers that were used in the Ponti 2020 research
X = df[['lymphocyte count', 'neutrophils count', 'Hypersensitive c-reactive protein', 'ESR', 'Interleukin 6', 'D-D dimer', 'NLR']]
y = df['outcome']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2)

# Fit decision tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Plot decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(dt_model, feature_names=X.columns, class_names=['Discharge', 'Dead'], filled=True)
plt.show()

# Initialize KFold
kf = KFold(n_splits=5, shuffle=True, random_state=41)

# Decision Tree
dt_scores = cross_val_score(dt_model, X_imputed, y, cv=kf)
print("Decision Tree Cross-validation scores:", dt_scores)
print("Decision Tree Mean Accuracy Ponti:", dt_scores.mean())