#%%
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import openpyxl
from pmdarima import auto_arima
from tqdm import tqdm

from statsmodels.tsa.x13 import x13_arima_analysis
from statsmodels.tsa.x13 import X13Warning



import warnings
# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)  # to avoid SettingWithCopyWarning




#%%
def main():
    
    df_export_ANALYSIS = pd.read_pickle("data/cleaned/total_export_FirstAnalysis.pkl")

    # target = df_export_ANALYSIS.columns.to_list()
    target = ['Exports,_f.o.b._(BOP_basis)',]
    display("len:", len(target))
    display(f"target: {target}")

    for name in tqdm(target,
                    desc="Processing",
                    dynamic_ncols=True):
        
        # if name in ['Exports,_f.o.b._(BOP_basis)', 'Timing_Adjustment']:
        #     continue
        
        print(f"Processing {name}...")
        col= name
        cols = [name]
        df = df_export_ANALYSIS[[name]]
        if name in ['Crude_oil', 'Photographic_&_cinematographic_instruments_&_supplies']:
            ###DATA IMPERFECTIONS
            df = df[df.index >= "1996-11-01"]
        df[name] = df[name].replace(0, np.nan).ffill().bfill()

        ts = df[name].dropna()
        '''
        additional columns previously not included in the analysis (Re-run)
        '''
        # X13-ARIMA-SEATS deseasonalization
        #DEsesonality using X13 by US Census Bureau

        # Suppress specific X13 warnings
        warnings.filterwarnings("ignore", category=X13Warning)

        x13_path='/home/wheelfredie/scripts/BoT_Exports/data/others/x13as'
        freq='M'

        dict_deseasonalized_value = {}
        dict_deseasonalized_pct_change = {}


        #deseasonalize the value 

        result = x13_arima_analysis(ts, 
                                x12path=x13_path, 
                                freq=freq)

        # Convert components to DataFrame
        df1 = pd.DataFrame({
                        'observed': result.observed,
                        'seasadj': result.seasadj,
                        'trend': result.trend,
                        'irregular': result.irregular,
                        'seasonal_factor': result.observed / result.seasadj
                    })
        # print(df1)
        dict_deseasonalized_value[name] = df1
            
        #convert dictionary to pickle and json
        with open(f'data/cleaned/deseasonalised_x13/update/dict_deseasonalized_value_{col}.pickle', 'wb') as handle:
            pickle.dump(dict_deseasonalized_value, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # #read the pickle file
        with open(f'data/cleaned/deseasonalised_x13/update/dict_deseasonalized_value_{name}.pickle', 'rb') as handle:
            dict_deseasonalized_value = pickle.load(handle)[name]
        adj_ts = dict_deseasonalized_value['seasadj'].pct_change().dropna()
            

            
        # try:
            
        #Walk Forward parameters
        window_size = 180  # 15 years of monthly data
        total_observations = len(adj_ts)  # Total data points
        steps = total_observations - window_size  # Remaining points after initial training window

        # Generate periods DataFrame
        periods_df = generate_periods_df(adj_ts, window_size, steps)
        periods_df = periods_df.iloc[[-1]]
        
        for col in periods_df.columns:
            periods_df[col] = periods_df[col].dt.to_timestamp()
        
        # display(periods_df)
        # ###
        # #FOR TESTING
        # periods_df = periods_df[:5] ###REMOVE----REMOVE---REMOVE--FOR--LIVE RUN####
        # ###

        # Perform walk-forward forecasting
        rolling_forecasts = walk_forward_forecasting(adj_ts, periods_df)
        
        forecast = pd.DataFrame(rolling_forecasts, columns=['forecast'])
        actual = adj_ts[adj_ts.index.isin(forecast.index)]

        df_accuracy = calculate_accuracy_metrics(actual, forecast)
        
        # display(df_accuracy)
        
        #get previous data
        with open(f'data/cleaned/SARIMA_RollWalkForward/combined/accuracy/df_accuracy_{name}.pickle',
                'rb') as handle:
            df_accuracy_prev = pickle.load(handle)
            
        with open(f'data/cleaned/SARIMA_RollWalkForward/combined/periods/periods_df_{name}.pickle',
                'rb') as handle:
            periods_df_prev = pickle.load(handle)
            
        #combine the previous and current data
        df_accuracy = pd.concat([df_accuracy_prev, df_accuracy])
        periods_df = pd.concat([periods_df_prev, periods_df])
        
        display(df_accuracy)
        display(periods_df)
        
        
        #save the df_accuracy & periods_df as pickle
        with open(f'data/cleaned/SARIMA_RollWalkForward/combined/updated/accuracy/df_accuracy_{name}.pickle',
                'wb') as handle:
            pickle.dump(df_accuracy, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(f'data/cleaned/SARIMA_RollWalkForward/combined/updated/periods/periods_df_{name}.pickle',
                'wb') as handle:
            pickle.dump(periods_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # except Exception as e:
        #     print(f"Error in {col}: {e}")
            

            
            
            
        
#%%
'''
HelperFunnctions
'''
# Generate periods DataFrame for walk-forward validation
def generate_periods_df(series, window_size, steps):
    if not isinstance(series.index, pd.PeriodIndex):
        series.index = pd.period_range(start=series.index[0], periods=len(series), freq='M')

    periods = []
    for t in range(steps):
        train_end_index = window_size + t
        periods.append({
            'window_start': series.index[0],
            'window_end': series.index[train_end_index - 1],
            'test': series.index[train_end_index]
        })
    
    # Add the last window
    periods.append({
        'window_start': series.index[0],
        'window_end': series.index[-1],
        # Test is not in the data, need to add next month
        'test': (series.index[-1] + 1)
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
            
            # Convert row["window_end"] to Period type that matches the series index
            window_end_period = row["window_end"].to_period(freq=series.index.freq)
                    
            train_data = series[series.index <= window_end_period]  # Include data up to the end of the window
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
# %%
