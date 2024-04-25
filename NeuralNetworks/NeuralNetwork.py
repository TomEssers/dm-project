import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def neural_network(data, biomarkers):

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

    # Define hyperparameters
    num_neurons_layer1 = 32
    num_neurons_layer2 = 64
    learning_rate = 0.009928553629858764
    batch_size = 32
    num_epochs = 50

    # Define K-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True)

    # Initialize lists to store evaluation metrics
    cv_accuracy = []
    cv_precision = []
    cv_recall = []
    cv_f1_score = []
    cv_auc = []

    # Initialize lists to store TP, FP, TN, FN
    cv_tp = []
    cv_fp = []
    cv_tn = []
    cv_fn = []

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

    # Perform K-fold cross-validation
    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model
        model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=0)

        # Predict probabilities
        y_pred_prob = model.predict(X_test, verbose=0)

        # Threshold probabilities to get predicted classes
        y_pred = (y_pred_prob > 0.5).astype(int)

        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Append confusion matrix components to lists
        cv_tp.append(tp)
        cv_fp.append(fp)
        cv_tn.append(tn)
        cv_fn.append(fn)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        # Append evaluation metrics to lists
        cv_accuracy.append(accuracy)
        cv_precision.append(precision)
        cv_recall.append(recall)
        cv_f1_score.append(f1)
        cv_auc.append(auc)

    # Calculate and print mean evaluation metrics
    mean_accuracy = np.mean(cv_accuracy)
    mean_precision = np.mean(cv_precision)
    mean_recall = np.mean(cv_recall)
    mean_f1_score = np.mean(cv_f1_score)
    mean_auc = np.mean(cv_auc)

    sum_tp = np.sum(cv_tp)
    sum_fp = np.sum(cv_fp)
    sum_tn = np.sum(cv_tn)
    sum_fn = np.sum(cv_fn)

    return mean_accuracy, mean_precision, mean_recall, mean_f1_score, mean_auc, sum_tp, sum_fp, sum_tn, sum_fn