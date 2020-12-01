'''
This script contains a set of functions that are model agnostic
'''

import pandas as pd
import numpy as np
import datetime
import time
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

    y_true = np.array(y_true_with_date.iloc[:,1])
    y_pred = np.array(y_prediction)


    # get the total error and the overall MAE

    overall_mae = mean_absolute_error(y_true, y_pred)
    absolute_error = overall_mae * len(y_true)

    print('\n----------------------------------------------')
    print(f'The absolute error (total actual minus  forecast) in MW is: {round(absolute_error, 2)}')
    print('----------------------------------------------\n')

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



# Cyclical encoding of the Date! For our usecase we suggest using:
# df = cyclical_encoder(df,1440,"minute")
# df = cyclical_encoder(df,12,"month")
# df = cyclical_encoder(df,7,"weekday")
# df["Year"] = df["Date"].dt.year-2014


def cyclical_encoder(df,T,time_period,Date_column="Date"):
    '''
    Take in a Datetime dataframe and return the same Dataframe that now includes the two cyclical encoded columns
    '''
    if time_period == "minutes":
        df[("sin_"+time_period)] = np.sin((df.loc[:,Date_column].dt.hour*60 + df.loc[:,Date_column].dt.minute) * 2*np.pi/T)
        df[("cos_"+time_period)] = np.sin((df.loc[:,Date_column].dt.hour*60 + df.loc[:,Date_column].dt.minute) * 2*np.pi/T)
    else:
        df[("sin_"+time_period)] = np.sin(getattr(df.loc[:,Date_column].dt,time_period) * 2*np.pi/T)
        df[("cos_"+time_period)] = np.cos(getattr(df.loc[:,Date_column].dt,time_period) * 2*np.pi/T)
    
    return df


## feature engineering for weather data



def create_feature_channels(df,stations,lat_steps,lon_steps,features,fillvalue=np.nan):
    '''
    This function will create "Pictures" where the pictures of the image correspond to the geographical location based on latitude and longitude 
    The value of each pixel corresponst to the value of the feature for that channel
    
    The input Parameters:
    ----------
    df : TYPE pandas dataframe
        df containing the features for different tables, named according to our convention
    stations : TYPE pandas dataframe
        a dataframe containing the id and latitude and longitude of the stations used
    lat_steps : TYPE int
        amouunt indicates the amount of pixels in the y-direction
    lon_steps : TYPE int
        amouunt indicates the amount of pixels in the x-direction
    features : TYPE list of strings
        a list of features we want to include as feature channels

    Returns
    -------
    a 4d tensor, the first 2 dimensions are the lat and lon, the 3rd dimension corresponds to the feature channels, and the fourth is time

    '''
    lat_range = [47.3,55]
    lon_range = [5.625,15]
    stepsize_lat = (lat_range[-1]-lat_range[0])/(lat_steps-1)
    stepsize_lon = (lon_range[-1]-lon_range[0])/(lon_steps-1)
    
    amount_of_features = len(features)    
    time_steps = df.shape[0]    
    feature_channels = np.zeros((lat_steps,lon_steps,amount_of_features,time_steps),"float32")*fillvalue()
    
    for i in range(lat_steps):
        for j in range(lon_steps):
            lat_sub = [lat_range[0]+stepsize_lat*i,lat_range[0]+stepsize_lat*(i+1)]
            lon_sub = [lon_range[0]+stepsize_lon*j,lon_range[0]+stepsize_lon*(j+1)]
            
    
            select_stations = (stations.latitude > lat_sub[0]) & (stations.latitude <= lat_sub[1]) \
                                &(stations.longitude > lon_sub[0]) & (stations.longitude <= lon_sub[1])
            station_string = ""
            for k in stations.loc[select_stations,"id"]:
                station_string += k +"|"
    
            for cnt,l in enumerate(features):
                if (len(station_string)>1):
                    if (df.filter(regex=station_string[:-1]).filter(regex=l).shape[1]>0):
                        feature_channels[i,j,cnt,:] = df.filter(regex=station_string[:-1]).filter(regex=l).mean(axis=1)

    
    return feature_channels

### feature engineering for the installed capacity

def create_feature_channels_installed_capacity(df,date_column,lat_steps,lon_steps,time_res,features=["Wind","Solar"],suppress=True):
    '''
    This function takes as a dataframe containing the installed capacity in germany and returns a grid with feature channels
    Also see create_feature_channels
    
    The input Parameters:
    ----------
    df : TYPE pandas dataframe
        Dataframe containing the installed capacity per type in Germany
    date_column : TYPE pandas series
        a pandas series with datetime objects containing the range of dates the feature channels should span
    lat_steps : TYPE int
        amouunt indicates the amount of pixels in the y-direction
    lon_steps : TYPE int
        amouunt indicates the amount of pixels in the x-direction
    time_res : TYPE int
        number of timesteps per day e.g. 96 for 15 min resolution
    features : TYPE, optional list of strings
        Gives the capacity types we want to create feature channels for
    suppress : TYPE, optional boolean
        set to True if print statements throughout the operation are wanted

    Returns
    -------
    feature_channels : numpy 4d tensor
        The feature channel "picture" of Germany

    '''
    df.loc[:,"commissioning_date"] = pd.to_datetime(df.loc[:,"commissioning_date"], format='%Y-%m-%d')
    df.loc[:,"decommissioning_date"].fillna("01.01.2055",inplace=True)
    df.loc[:,"decommissioning_date"] = pd.to_datetime(df.loc[:,"decommissioning_date"], format='%d.%m.%Y')
    
    time_steps = date_column.shape[0]
    amount_of_features = len(features)
    
    lat_range = [47.3,55]
    lon_range = [5.625,15]
    
    stepsize_lat = (lat_range[-1]-lat_range[0])/(lat_steps-1)
    stepsize_lon = (lon_range[-1]-lon_range[0])/(lon_steps-1)
    

    feature_channels = np.zeros((lat_steps,lon_steps,amount_of_features,time_steps),"float32")
    
    start = time.time()




    df_dict = {}
    
    for i in range(lat_steps):
        start_loop = time.time()
        df_dict[i] = {}
        for j in range(lon_steps):
            df_dict[i][j] = {}
            lat_sub = [lat_range[0]+stepsize_lat*i,lat_range[0]+stepsize_lat*(i+1)]
            lon_sub = [lon_range[0]+stepsize_lon*j,lon_range[0]+stepsize_lon*(j+1)]
    
            for cnt,l in enumerate(features):
                select_stations = (df.lat > lat_sub[0]) & (df.lat <= lat_sub[1]) \
                                  &(df.lon > lon_sub[0]) & (df.lon <= lon_sub[1]) \
                                  &(df.energy_source_level_2 == l)
    
                df_dict[i][j][l] = df.loc[select_stations,["electrical_capacity","commissioning_date","decommissioning_date"]]
        end_loop = time.time()
        if not suppress:
            print(f"The loop took {round(end_loop - start_loop,2)}s step {i+1} out of {lat_steps}")
    if not suppress:                
        end = time.time()
        print(f"Creating the dictionary took {round(end - start,2)}s")
    
    for t in range(int(time_steps/time_res)-1):
        start_loop = time.time()
        date = date_column.iloc[time_res*t]
        for i in range(lat_steps):
            for j in range(lon_steps):
                for cnt,l in enumerate(features):
                    if df_dict[i][j][l].shape[0] > 0:
                        select_stations = (df_dict[i][j][l].commissioning_date <= date) \
                                           & (df_dict[i][j][l].decommissioning_date > date) 
                        feature_channels[i,j,cnt,time_res*t:time_res*(t+1)] = df_dict[i][j][l].loc[select_stations,"electrical_capacity"].sum()
                    else:
                        feature_channels[i,j,cnt,time_res*t:time_res*(t+1)] = 0
        if not suppress:
            end_loop = time.time()
            print(f"the loop took{round(end_loop - start_loop,2)} , step {t} of {int(time_steps/time_res)-1}")

    if not suppress:
        end = time.time()
        print(f"Creating the feature channels took {round(end - start,2)}s")
    
    return feature_channels
