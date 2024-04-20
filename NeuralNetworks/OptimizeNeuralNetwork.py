import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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

X = data[['Lactate dehydrogenase', 'Hypersensitive c-reactive protein', '(%)lymphocyte']]
y = data['outcome']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Define search space for hyperparameters
space = {
    'num_neurons_layer1': hp.choice('num_neurons_layer1', [16, 32, 64, 128]),
    'num_neurons_layer2': hp.choice('num_neurons_layer2', [16, 32, 64, 128]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
    'batch_size': hp.choice('batch_size', [16, 32, 64]),
    'num_epochs': hp.choice('num_epochs', [50, 100, 150])
}

# Define objective function to minimize
def objective(params):
    num_neurons_layer1 = params['num_neurons_layer1']
    num_neurons_layer2 = params['num_neurons_layer2']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']

    # Build the neural network model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_scaled.shape[1],)),
        tf.keras.layers.Dense(num_neurons_layer1, activation='relu'),
        tf.keras.layers.Dense(num_neurons_layer2, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(X_scaled, y, batch_size=batch_size, epochs=num_epochs, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_scaled, y, verbose=0)

    return {'loss': -accuracy, 'status': STATUS_OK}

# Perform hyperparameter optimization
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=50, trials=trials)

print("Best hyperparameters:", best)