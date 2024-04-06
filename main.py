from NeuralNetworks.NeuralNetwork import neural_network
from XgBoostDecisionTree.XgBoostDecisionTree import xgboost
import pandas as pd
import numpy as np

# Load the dataset
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

# Get all Neural Network values
neural_network_ponti_values = neural_network(data=data, biomarkers=['lymphocyte count', 'neutrophils count', 
                                                                   'Hypersensitive c-reactive protein', 'ESR', 'Interleukin 6', 'D-D dimer', 'NLR'])

neural_network_semiz_values = neural_network(data=data, biomarkers=['Hypersensitive c-reactive protein', 'procalcitonin', 'Interleukin 6',
        'lymphocyte count', 'neutrophils count', 'D-D dimer', 'ferritin', 'Red blood cell distribution width ',
        'aspartate aminotransferase', 'glutamic-pyruvic transaminase', 'Total bilirubin', 'albumin', 'NLR'])

neural_network_zhao_values = neural_network(data=data, biomarkers=['Lactate dehydrogenase', 'Hypersensitive c-reactive protein', '(%)lymphocyte'])

# Get all XgBoost (decision trees) values
xgboost_ponti_values = xgboost(data=data, biomarkers=['lymphocyte count', 'neutrophils count', 
                                                                   'Hypersensitive c-reactive protein', 'ESR', 'Interleukin 6', 'D-D dimer', 'NLR'])

xgboost_semiz_values = xgboost(data=data, biomarkers=['Hypersensitive c-reactive protein', 'procalcitonin', 'Interleukin 6',
        'lymphocyte count', 'neutrophils count', 'D-D dimer', 'ferritin', 'Red blood cell distribution width ',
        'aspartate aminotransferase', 'glutamic-pyruvic transaminase', 'Total bilirubin', 'albumin', 'NLR'])

xgboost_zhao_values = xgboost(data=data, biomarkers=['Lactate dehydrogenase', 'Hypersensitive c-reactive protein', '(%)lymphocyte'])

print("ponti_neural_network:")
print(neural_network_ponti_values)
print("semiz_neural_network:")
print(neural_network_semiz_values)
print("zhao_neural_network:")
print(neural_network_zhao_values)
print("ponti_xgboost:")
print(xgboost_ponti_values)
print("semiz_xgboost:")
print(xgboost_semiz_values)
print("zhao_xgboost:")
print(xgboost_zhao_values)