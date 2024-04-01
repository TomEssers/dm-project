import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.impute import SimpleImputer

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

# Split features and target
X = data[['Lactate dehydrogenase', 'Hypersensitive c-reactive protein', '(%)lymphocyte']]
y = data['outcome']

# Reset index of y to match X
y = y.reset_index(drop=True)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Define hyperparameters
num_neurons_layer1 = 64
num_neurons_layer2 = 32
learning_rate = 0.005
batch_size = 30
num_epochs = 50

# Define K-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize list to store cross-validation scores
cv_scores = []

# Perform K-fold cross-validation
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Build the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(num_neurons_layer1, activation='elu', input_shape=(3,)),
        tf.keras.layers.Dense(num_neurons_layer2, activation='elu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=0)

    # Evaluate the model
    _, test_acc = model.evaluate(X_test, y_test)
    cv_scores.append(test_acc)

# Calculate and print mean cross-validation score
mean_cv_score = np.mean(cv_scores)
print('Mean cross-validation accuracy Neural Network Zhao:', mean_cv_score)