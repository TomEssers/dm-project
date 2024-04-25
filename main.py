import pandas as pd
import numpy as np
from DecisionTrees.DecisionTree import decision_tree
from NeuralNetworks.NeuralNetwork import neural_network
from XgBoostDecisionTree.XgBoostDecisionTree import xgboost
from LogisticRegressions.LogisticRegression import logistic_regression
from KNN.KNN import knn


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


def run_decisiontrees(data, biomarkers, dataframe, biomarker_name):
    # Run the decision tree model
    results = decision_tree(data=data, biomarkers=biomarkers)
    # Add values to dataframe
    dataframe.loc[len(dataframe)] = ['decision_tree', biomarker_name, *results]


def run_logistic_regressions(data, biomarkers, dataframe, biomarker_name):
    # Run the logistic regression model
    results = logistic_regression(data=data, biomarkers=biomarkers)
    # Add values to dataframe
    dataframe.loc[len(dataframe)] = ['logistic_regression', biomarker_name, *results]


def run_knn(data, biomarkers, knn_k_amount, dataframe, biomarker_name):
    # Run the KNN model
    results = knn(data=data, biomarkers=biomarkers, knn_k_amount=knn_k_amount)
    # Add values to dataframe
    dataframe.loc[len(dataframe)] = ['knn', biomarker_name, *results]


def run_neural_networks(data, biomarkers, dataframe, biomarker_name):
    # Run the neural network model
    results = neural_network(data=data, biomarkers=biomarkers)
    # Add values to dataframe
    dataframe.loc[len(dataframe)] = ['neural_network', biomarker_name, *results]


def run_xgboosts(data, biomarkers, dataframe, biomarker_name):
    # Run the xgboost model
    results = xgboost(data=data, biomarkers=biomarkers)
    # Add values to dataframe
    dataframe.loc[len(dataframe)] = ['xgboost', biomarker_name, *results]


def main():
    # Set location of the data
    data_location = "data/time_series_375_preprocess_en.csv"

    # Prepare the data for the DM models
    data = prepare_data(path=data_location)

    # Set the sets of biomarkers
    # These are taken from 3 different researches, namely Ponti (2020), Semiz (2022), and Zhao (2020)
    ponti = ['lymphocyte count', 'neutrophils count', 'Hypersensitive c-reactive protein', 'ESR', 'Interleukin 6',
             'D-D dimer', 'NLR']
    semiz = ['Hypersensitive c-reactive protein', 'procalcitonin', 'Interleukin 6',
             'lymphocyte count', 'neutrophils count', 'D-D dimer', 'ferritin', 'Red blood cell distribution width ',
             'aspartate aminotransferase', 'glutamic-pyruvic transaminase', 'Total bilirubin', 'albumin', 'NLR']
    zhao = ['Lactate dehydrogenase', 'Hypersensitive c-reactive protein', '(%)lymphocyte']

    # Create a new DataFrame for the results
    results_df = pd.DataFrame(columns=['model_name', 'biomarker_set', 'accuracy', 'precision', 'recall', 'f1', 'AUC'])

    # Perform all of the data mining methods for each of the biomarker set:
    # PONTI:
    for i in range(100):
        run_decisiontrees(data=data, biomarkers=ponti, dataframe=results_df, biomarker_name='ponti')
        run_logistic_regressions(data=data, biomarkers=ponti, dataframe=results_df, biomarker_name='ponti')
        run_neural_networks(data=data, biomarkers=ponti, dataframe=results_df, biomarker_name='ponti')
        run_xgboosts(data=data, biomarkers=ponti, dataframe=results_df, biomarker_name='ponti')
        run_knn(data=data, biomarkers=ponti, knn_k_amount=17, dataframe=results_df, biomarker_name='ponti')

        # SEMIZ
        run_decisiontrees(data=data, biomarkers=semiz, dataframe=results_df, biomarker_name='semiz')
        run_logistic_regressions(data=data, biomarkers=semiz, dataframe=results_df, biomarker_name='semiz')
        run_neural_networks(data=data, biomarkers=semiz, dataframe=results_df, biomarker_name='semiz')
        run_xgboosts(data=data, biomarkers=semiz, dataframe=results_df, biomarker_name='semiz')
        run_knn(data=data, biomarkers=semiz, knn_k_amount=7, dataframe=results_df, biomarker_name='semiz')

        # ZHAO
        run_decisiontrees(data=data, biomarkers=zhao, dataframe=results_df, biomarker_name='zhao')
        run_logistic_regressions(data=data, biomarkers=zhao, dataframe=results_df, biomarker_name='zhao')
        run_neural_networks(data=data, biomarkers=zhao, dataframe=results_df, biomarker_name='zhao')
        run_xgboosts(data=data, biomarkers=zhao, dataframe=results_df, biomarker_name='zhao')
        run_knn(data=data, biomarkers=zhao, knn_k_amount=13, dataframe=results_df, biomarker_name='zhao')
        
    # Place the results in a CSV file
    results_df.to_csv('results2.csv', index=False)


if __name__ == "__main__":
    main()
