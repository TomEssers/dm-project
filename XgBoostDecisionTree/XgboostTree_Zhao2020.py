import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
import xgboost as xgb

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

# Get only the three columns that were used in the initial research
X = df[['Lactate dehydrogenase', 'Hypersensitive c-reactive protein', '(%)lymphocyte']]
y = df['outcome']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Initialize XGBoost model
xgb_model = xgb.XGBClassifier()

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation using StratifiedKFold
xgb_scores = cross_val_score(xgb_model, X_imputed, y, cv=skf)
print("XGBoost Cross-validation scores:", xgb_scores)
print("XGBoost Mean Accuracy:", xgb_scores.mean())
