'''
This script contains a set of functions that are model agnostic
'''

import pandas as pd
import numpy as np

def train_val_test_split(df, target_vars, val_days, test_days):
    '''
    Split a time series data set into train, validation and test set
    INPUT:
    - df: dataset in pandas dataframe form
    - target_vars: list of the target/label variables
    - val_days: integer representing number of days to use for the validation set
    - test_days: integer representing number of days to use for the test set

    OUTPUT:
    - X_train: data set for training
    - y_train: vector of the target for the training of the model
    - X_val: data set for validation
    - y_val: vector of the target values for the validation of the model
    - X_test: data set to test the model
    - y_val: vector of the target values for testing the model
    '''

    print('The shape of the data set is: {}'.format(df.shape))

    rows_for_test = test_days*96
    rows_for_val = val_days*96

    df_features = df.drop(columns=target_vars).copy()
    df_target = df[target_vars].copy()

    X_train = df_features.iloc[:(df_features.shape[0]-rows_for_test-rows_for_val)]
    y_train = df_target.iloc[:(df_target.shape[0]-rows_for_test-rows_for_val)]
    print('\n--------------------------------------------')
    print('The shape of the train set is: {}'.format(X_train.shape))
    print('The shape of the target variable is: {}'.format(y_train.shape))
    print('--------------------------------------------')

    X_val = df_features.iloc[X_train.shape[0]:(df_features.shape[0]-rows_for_test)]
    y_val = df_target.iloc[y_train.shape[0]:(df_target.shape[0]-rows_for_test)]
    print('\n--------------------------------------------')
    print('The shape of the validation set is: {}'.format(X_val.shape))
    print('The shape of the target variable for the validation set is: {}'.format(y_val.shape))
    print('--------------------------------------------')

    X_test = df_features.iloc[X_val.shape[0]+X_train.shape[0]:]
    y_test = df_target.iloc[y_val.shape[0]+X_train.shape[0]:]
    print('\n--------------------------------------------')
    print('The shape of the test set is: {}'.format(X_test.shape))
    print('The shape of the target variable for the test set is: {}'.format(y_test.shape))
    print('--------------------------------------------')

    return X_train, y_train, X_val, y_val, X_test, y_test

'''
def model_metrics(y_prediction, y_true):
    give me the mae of my model and compare it to entso-e and maybe the baseline - in best case use a graph
'''