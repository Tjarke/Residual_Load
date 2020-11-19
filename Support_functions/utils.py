'''
This script contains a set of functions that are model agnostic
'''

import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import mean_absolute_error

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

    target_vars_for_test = target_vars.copy()
    target_vars_for_test.append('Date')

    df_features = df.drop(columns=target_vars_for_test).copy()
    df_target = df[target_vars].copy()
    df_target_test = df[target_vars_for_test].copy()

    del df

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
    y_test = df_target_test.iloc[y_val.shape[0]+X_train.shape[0]:]
    print('\n--------------------------------------------')
    print('The shape of the test set is: {}'.format(X_test.shape))
    print('The shape of the target variable for the test set is: {}'.format(y_test.shape))
    print('--------------------------------------------')

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_model_metrics(y_true_with_date, y_prediction):

    # get the total error in MW

    y_true = np.array(y_true_with_date.iloc[:,1])
    y_pred = np.array(y_prediction)

    total_pred = sum(y_pred)
    total_true = sum(y_true)
    absolute_error = total_true - total_pred

    print('\n----------------------------------------------')
    print(f'The absolute error (total actual minus  forecast) in MW is: {round(absolute_error, 2)}')
    print('----------------------------------------------\n')

    # get the overall MAE

    overall_mae = mean_absolute_error(y_true, y_pred)

    print('\n----------------------------------------------')
    print(f'The overall mean absolute error of the model in MW is: {overall_mae}')
    print('----------------------------------------------\n')

    # get the overall mean absolute scaled error (MASE)

    naive_forecast = y_true[1:]
    y_true_for_mase = y_true[:-1]
    mae_naive = mean_absolute_error(y_true_for_mase, naive_forecast)
    overall_mae_without_first_observation = mean_absolute_error(y_true[1:], y_pred[1:])

    overall_mase = overall_mae_without_first_observation/mae_naive

    print('\n----------------------------------------------')
    print(f'The overall mean absolute scaled error of the model in MW is: {overall_mase}')
    print('Please note: to calculate the MASE, the prediction for the first observation was omitted')
    print('----------------------------------------------\n')

    # get the MAE for every day and return a dataframe and charts

    time_stamp = np.array(
                 pd.to_datetime(y_true_with_date.iloc[:,0], format='%Y-%m-%d %H:%M:%S').dt.date
                 )

    date_list = list()
    y_true_list = list()
    y_pred_list = list()

    df = pd.DataFrame({'Date': time_stamp,
                       'y_true': y_true,
                       'y_pred': y_pred})

    list_of_days = sorted(list(set(df.Date.values)))

    for day in list_of_days:
        sub_df = df.query('Date == @day')
        date_list.append(sub_df.Date.values)
        y_true_list.append(sub_df.y_true.values)
        y_pred_list.append(sub_df.y_pred.values)

    mae_list = list()
    for i, j, k in zip(y_true_list, y_pred_list, date_list):
        MAE = mean_absolute_error(i, j)
        mae_list.append(MAE)

    del df

    df = pd.DataFrame({'day':list_of_days, 'MAE': mae_list},)

    print('\n----------------------------------------------')
    print('This function also returns a dataframe with the MAE for each day')
    print('----------------------------------------------\n')

    return df


def iteratively_predict(model,X_test,n,name_of_naive_column="naive_System total load in MAW",suppress=True):
    '''
    This function will iteratively predict the next n timesteps for the complete test set. The test data should include a native prediction column.
    The column will be overwritten in order to predict new values.
    
    input: the model that predicts
           the test_features X_test
           number of timesteps to predict n
           name_of_naive_column= column containing the target with a 1 timestep lag
           
    output:
        predictions of the model
    '''
    
    X_test.reset_index(drop=True,inplace=True)
    
    y_p = np.array([])
    for i in range(int(X_test.shape[0])-1):
        if i%n == 0:
            prediction = model.predict(X_test.iloc[i,:].to_numpy().reshape(1, -1))
            y_p = np.append(y_p,prediction)
            if not suppress:
                print(f"predicted time period {i/n+1}")
        else:
            prediction = model.predict(X_test.iloc[i,:].to_numpy().reshape(1, -1))
            X_test.loc[i+1,name_of_naive_column] = prediction
            y_p = np.append(y_p,prediction)
    return y_p
