#%%
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import openpyxl
from pmdarima import auto_arima
from tqdm import tqdm

import warnings
# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)  # to avoid SettingWithCopyWarning




#%%
def main():
    #load the pickle file
    # with open('data/cleaned/deseasonalised_x13/dict_deseasonalized_value.pickle', 'rb') as handle:
    #     dict_deseasonalized_value = pickle.load(handle)

    # Iterate over the dictionary for each time series for the given cols
    while True:
        col_index = int(input("Enter the column index: (0,1,...,5)"))
        if col_index in range(6):
            break
        
    if col_index == 5:
        with open('data/cleaned/deseasonalised_x13/dict_deseasonalized_value_EXTRA.pickle', 'rb') as handle:
            dict_deseasonalized_value = pickle.load(handle)
        cols = ['Agriculture','Fishery', 'Mining', 'Manufacturing']
        print(f"Processing the following columns: {cols}")
        
    else:
        with open('data/cleaned/deseasonalised_x13/dict_deseasonalized_value.pickle', 'rb') as handle:
            dict_deseasonalized_value = pickle.load(handle)
            
        with open('data/cleaned/deseasonalised_x13/list_of_lists.pickle', 'rb') as handle:
            cols = pickle.load(handle)[col_index]
            print(f"Processing the following columns: {cols}")
        
    
        
    for col in tqdm(cols,
                    desc="Outer Columns"):
        print(f"Currently Processing: {col}")
        
        try:
            adj_ts = dict_deseasonalized_value[col]['seasadj'].pct_change().dropna()
            
            #Walk Forward parameters
            window_size = 180  # 15 years of monthly data
            total_observations = len(adj_ts)  # Total data points
            steps = total_observations - window_size  # Remaining points after initial training window

            # Generate periods DataFrame
            periods_df = generate_periods_df(adj_ts, window_size, steps)

            # ###
            # #FOR TESTING
            # periods_df = periods_df[:5] ###REMOVE----REMOVE---REMOVE--FOR--LIVE RUN####
            # ###

            # Perform walk-forward forecasting
            rolling_forecasts = walk_forward_forecasting(adj_ts, periods_df)
            
            forecast = pd.DataFrame(rolling_forecasts, columns=['forecast'])
            actual = adj_ts[adj_ts.index.isin(forecast.index)]

            df_accuracy = calculate_accuracy_metrics(actual, forecast)
            
            
            #save the df_accuracy & periods_df as pickle
            with open(f'data/cleaned/SARIMA_RollWalkForward/accuracy/df_accuracy_{col}.pickle',
                    'wb') as handle:
                pickle.dump(df_accuracy, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
            with open(f'data/cleaned/SARIMA_RollWalkForward/periods/periods_df_{col}.pickle',
                    'wb') as handle:
                pickle.dump(periods_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        except Exception as e:
            print(f"Error in {col}: {e}")
            continue
        
        # ###
        # #FOR TESTING
        # break ###REMOVE----REMOVE---REMOVE--FOR--LIVE RUN####
        # ###
        
        
        
        
#%%
'''
HelperFunnctions
'''
# Function to generate DataFrame of periods for walk-forward analysis
def generate_periods_df(series, window_size, steps):
    periods = []
    for t in range(steps):
        train_end_index = window_size + t
        periods.append({
            'window_start': series.index[0],
            'window_end': series.index[train_end_index - 1],
            'test': series.index[train_end_index]
        })
    return pd.DataFrame(periods)

# Function to perform walk-forward forecasting using the periods DataFrame
def walk_forward_forecasting(series, periods_df):
    predictions = []
    date_lst = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # Ignore FutureWarnings
        for _, row in tqdm(periods_df.iterrows(),
                           total=len(periods_df),
                           desc="Walk-Forward Forecasting",
                           dynamic_ncols=True):
            train_data = series[series.index <= row["window_end"]]  # Include data up to the end of the window
            model = auto_arima(train_data, seasonal=True, m=12, trace=False,
                            error_action='ignore', suppress_warnings=True)
            # Forecast the next period
            forecast = model.predict(n_periods=1)
            predictions.append(forecast[0])
            date_lst.append(row["test"])

    
    forecast_series = pd.Series(predictions, index=date_lst)
    return forecast_series

#Function to return df of actuals and predictions with error metrics
def calculate_accuracy_metrics(actual, forecast):
    """
    Calculate accuracy metrics for the actual and forecasted values.

    Parameters:
    actual (pd.Series): Series of actual values.
    forecast (pd.Series): Series of forecasted values.

    Returns:
    pd.DataFrame: DataFrame containing actual, forecast, and calculated metrics.
    """
    df_accuracy = pd.concat([actual, forecast], axis=1)
    df_accuracy.columns = ['actual', 'forecast']
    df_accuracy['error'] = df_accuracy['actual'] - df_accuracy['forecast']
    df_accuracy['abs_error'] = abs(df_accuracy['error'])
    df_accuracy['squared_error'] = df_accuracy['error']**2
    df_accuracy['abs_percentage_error'] = abs(df_accuracy['error'] / df_accuracy['actual'])
    
    return df_accuracy




#%%
if __name__ == '__main__':
    main()