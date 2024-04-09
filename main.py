from NeuralNetworks.NeuralNetwork import neural_network
from XgBoostDecisionTree.XgBoostDecisionTree import xgboost
from LogisticRegressions.LogisticRegression import logistic_regression
from KNN.KNN import knn
import pandas as pd
import numpy as np


def prepare_data(path):
        # Load the dataset
        data = pd.read_csv(path, delimiter=';')

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

        # Return the prepared dataset
        return data


def run_logistic_regressions(data, csv_path):
        
        # Run logistic regression 3 times
        logistic_regression_ponti_values = logistic_regression(data=data, biomarkers=['lymphocyte count', 'neutrophils count', 
                                                                        'Hypersensitive c-reactive protein', 'ESR', 'Interleukin 6', 'D-D dimer', 'NLR'])

        logistic_regression_semiz_values = logistic_regression(data=data, biomarkers=['Hypersensitive c-reactive protein', 'procalcitonin', 'Interleukin 6',
                'lymphocyte count', 'neutrophils count', 'D-D dimer', 'ferritin', 'Red blood cell distribution width ',
                'aspartate aminotransferase', 'glutamic-pyruvic transaminase', 'Total bilirubin', 'albumin', 'NLR'])

        logistic_regression_zhao_values = logistic_regression(data=data, biomarkers=['Lactate dehydrogenase', 'Hypersensitive c-reactive protein', '(%)lymphocyte'])

        logistic_regression_all_values = logistic_regression(data=data, biomarkers="all")

        # To be removedp print results
        print("ponti lr:")
        print(logistic_regression_ponti_values)
        print("semiz lr:")
        print(logistic_regression_semiz_values)
        print("zhao lr:")
        print(logistic_regression_zhao_values)
        print("all lr:")
        print(logistic_regression_all_values)

        # Save the results to the CSV

def run_knn(data, csv_path):

        # Run logistic regression 3 times
        knn_ponti_values = knn(data=data, biomarkers=['lymphocyte count', 'neutrophils count', 
                                                                        'Hypersensitive c-reactive protein', 'ESR', 'Interleukin 6', 'D-D dimer', 'NLR'])

        knn_semiz_values = knn(data=data, biomarkers=['Hypersensitive c-reactive protein', 'procalcitonin', 'Interleukin 6',
                'lymphocyte count', 'neutrophils count', 'D-D dimer', 'ferritin', 'Red blood cell distribution width ',
                'aspartate aminotransferase', 'glutamic-pyruvic transaminase', 'Total bilirubin', 'albumin', 'NLR'])

        knn_zhao_values = knn(data=data, biomarkers=['Lactate dehydrogenase', 'Hypersensitive c-reactive protein', '(%)lymphocyte'])

        knn_all_values = knn(data=data, biomarkers="all")

        # To be removed print results
        print("ponti knn:")
        print(knn_ponti_values)
        print("semiz knn:")
        print(knn_semiz_values)
        print("zhao knn:")
        print(knn_zhao_values)
        print("all knn:")
        print(knn_all_values)

        # Save the results to the CSV

def run_neural_networks(data, csv_path):
        # Get all Logistic Regression values
        # Get all Neural Network values
        neural_network_ponti_values = neural_network(data=data, biomarkers=['lymphocyte count', 'neutrophils count', 
                                                                        'Hypersensitive c-reactive protein', 'ESR', 'Interleukin 6', 'D-D dimer', 'NLR'])

        neural_network_semiz_values = neural_network(data=data, biomarkers=['Hypersensitive c-reactive protein', 'procalcitonin', 'Interleukin 6',
                'lymphocyte count', 'neutrophils count', 'D-D dimer', 'ferritin', 'Red blood cell distribution width ',
                'aspartate aminotransferase', 'glutamic-pyruvic transaminase', 'Total bilirubin', 'albumin', 'NLR'])

        neural_network_zhao_values = neural_network(data=data, biomarkers=['Lactate dehydrogenase', 'Hypersensitive c-reactive protein', '(%)lymphocyte'])

        neural_network_all_values = neural_network(data=data, biomarkers="all")

        print("ponti_neural_network:")
        print(neural_network_ponti_values)
        print("semiz_neural_network:")
        print(neural_network_semiz_values)
        print("zhao_neural_network:")
        print(neural_network_zhao_values)
        print("all_neural_network:")
        print(neural_network_all_values)


def run_xgboosts(data, csv_path):
         # Get all XgBoost (decision trees) values
        xgboost_ponti_values = xgboost(data=data, biomarkers=['lymphocyte count', 'neutrophils count', 
                                                                        'Hypersensitive c-reactive protein', 'ESR', 'Interleukin 6', 'D-D dimer', 'NLR'])

        xgboost_semiz_values = xgboost(data=data, biomarkers=['Hypersensitive c-reactive protein', 'procalcitonin', 'Interleukin 6',
                'lymphocyte count', 'neutrophils count', 'D-D dimer', 'ferritin', 'Red blood cell distribution width ',
                'aspartate aminotransferase', 'glutamic-pyruvic transaminase', 'Total bilirubin', 'albumin', 'NLR'])

        xgboost_zhao_values = xgboost(data=data, biomarkers=['Lactate dehydrogenase', 'Hypersensitive c-reactive protein', '(%)lymphocyte'])

        xgboost_all_values = xgboost(data=data, biomarkers="all")
        
        print("ponti_xgboost:")
        print(xgboost_ponti_values)
        print("semiz_xgboost:")
        print(xgboost_semiz_values)
        print("zhao_xgboost:")
        print(xgboost_zhao_values)
        print("all_xgboost:")
        print(xgboost_all_values)


if __name__ == "__main__":
        
        # Set location of the data
        data_location = "data/time_series_375_preprocess_en.csv"

        # Prepare the data for the DM models
        data = prepare_data(path=data_location)
        
        # Create a new CSV file for the results
        
        Perform all of the data mining methods
        run_logistic_regressions(data=data, csv_path=None)
        run_neural_networks(data=data, csv_path=None)
        run_xgboosts(data=data, csv_path=None)
        run_knn(data=data, csv_path=None)
