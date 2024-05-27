import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import openpyxl
import warnings
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima
from tqdm import tqdm
from statsmodels.tsa.x13 import x13_arima_analysis, X13Warning
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import AutoReg
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import ttest_ind
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)  # to avoid SettingWithCopyWarning

# Stationarity: Augmented Dickey-Fuller Test
def adf_test(series):
    '''
    The Dickey-Fuller test is a unit root test that examines whether a unit root is present in a time series dataset. 
    A unit root implies that the series is non-stationary.
    The null hypothesis of the Dickey-Fuller test is that the time series possesses a unit root and is hence non-stationary. 
    The alternate hypothesis is that the time series is stationary.
    
    ADF test does not explicitly distinguish between deterministic trends and other forms of non-stationarity, 
    such as stochastic trends or cycles.
    '''
    result = adfuller(series)
    adf_stat = result[0]
    p_value = result[1]
    is_stationary = p_value < 0.05
    print('ADF Statistic:', adf_stat)
    print('p-value:', p_value)
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')
    return is_stationary

# Stationarity: KPSS Test
def kpss_test(series):
    '''
    The KPSS test is used to check for the presence of a unit root in a time series dataset, 
    but with the opposite null and alternate hypotheses compared to the Dickey-Fuller test.
    The null hypothesis of the KPSS test is that the time series is stationary around a deterministic trend. 
    The alternate hypothesis is that the series has a unit root and is hence non-stationary.
    
    Deterministic trend represents a systematic, predictable pattern of change, the trend component
    (Less tight bound compared to ADF test, but still indicates information value for statistical analysis)
    Properties of a deterministic trend:
    - predictability: Unlike random fluctuations, deterministic trends follow a clear pattern or function, making them predictable over time.
    - Stability: The trend remains consistent over the entire time period covered by the data without significant deviations.
    - Non-Stochastic: Deterministic trends are not influenced by random shocks or external factors; instead, they are driven by underlying systematic forces.
    '''
    result = kpss(series, regression='c')
    kpss_stat = result[0]
    p_value = result[1]
    is_stationary = p_value >= 0.05
    print('KPSS Statistic:', kpss_stat)
    print('p-value:', p_value)
    for key, value in result[3].items():
        print(f'Critical Value {key}: {value}')
    return is_stationary

# Plot ACF and PACF
def plot_acf_pacf(series, lags=24):
    '''
    Autocorrelation Function (ACF): Measures the correlation of the series with its lagged values.
    
    Partial Autocorrelation Function (PACF): Measures the correlation of the series with its lagged values,
    controlling for the values of the intervening lags.
    '''
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    
    # ACF plot
    acf_plot = plot_acf(series, lags=lags, ax=ax[0])
    ax[0].set_xticks(range(lags+1))
    ax[0].set_xticklabels(range(lags+1))
    
    # PACF plot
    pacf_plot = plot_pacf(series, lags=lags, ax=ax[1])
    ax[1].set_xticks(range(lags+1))
    ax[1].set_xticklabels(range(lags+1))
    
    plt.show()

# Decomposition
def decompose(series, model='additive', freq=12):
    decomposition = sm.tsa.seasonal_decompose(series, model=model, period=freq)
    decomposition.plot()
    plt.show()
    return decomposition

# Homoscedasticity Check: Plot Residuals
def plot_residuals(series):
    decomposition = sm.tsa.seasonal_decompose(series, model='additive')
    residual = decomposition.resid
    plt.plot(residual)
    plt.title('Residuals')
    plt.show()

# Overall Check Function
def check_time_series_assumptions(series, freq=12):
    print("Running ADF Test for Stationarity...")
    adf_result = adf_test(series)
    print(f"is_stationary: {adf_result}")
    
    print("\nRunning KPSS Test for Stationarity...")
    kpss_result = kpss_test(series)
    print(f"is_stationary: {kpss_result}")
    
    print("\nDecomposing the Series...")
    decomposition = decompose(series, freq=freq)
    
    print("\nPlotting Residuals for Homoscedasticity Check...")
    plot_residuals(series)
    
    #plot_acf_pacf
    plot_acf_pacf(series)
    
    assumptions_met = adf_result and kpss_result
    if assumptions_met:
        print("\nTime series assumptions are met.")
    else:
        print("\nTime series assumptions are not met. Further investigation needed.")
    
    return assumptions_met


# Function to generate DataFrame of periods for walk-forward analysis
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

def plot_actual_vs_forecast(df_accuracy, col_name, type="% Change"):
    """
    Plot the actual vs forecasted values and absolute errors, with R² and average absolute error in the title.

    Parameters:
    df_accuracy (pd.DataFrame): DataFrame containing actual, forecast, and calculated metrics.
    """
    colors = ['red', 'black', 'blue']
    fig, ax = plt.subplots(figsize=(30, 15))

    # Calculate R² and average absolute error
    r2 = r2_score(df_accuracy['actual'][:-1], df_accuracy['forecast'][:-1])
    average_abs_error = df_accuracy["abs_error"].mean()

    # Plot actual and forecasted values with alternating colors
    for i, (idx, row) in enumerate(df_accuracy.iterrows()):
        color = colors[i % len(colors)]  # Alternating between green, red, and blue
        ax.plot(idx, row['actual'], marker='o', markersize=10, color=color, label='Actual' if i == 0 else None)
        ax.plot(idx, row['forecast'], marker='^', markersize=10, color=color, label='Forecasted' if i == 0 else None)
        
    #Highlight the LAST forecasted values
    ax.plot(df_accuracy['forecast'].index[-1], df_accuracy['forecast'].iloc[-1], 
                marker='^', alpha=1, color='green', markersize=20, label='April 2024 Forecast')

    # Plot the time series line with the same color scheme
    df_accuracy["actual"].plot(ax=ax, color='black', linewidth=2, label='Time Series')

    # Set x-axis ticks to ensure all x-values are listed
    ax.set_xticks(df_accuracy.index)
    ax.set_xticklabels(df_accuracy.index)

    # Rotate x-axis labels for better readability and increase font size
    plt.xticks(rotation=90, fontsize=9)

    # Title with R² and average absolute error
    plt.title(f"Actual vs. Forecasted ({col_name}, {type})\nR²: {r2:.2f}, Mean Abs Error: {average_abs_error:.2f}")
    # Add legend
    plt.legend()

    # Create a subplot for absolute error
    fig, ax_err = plt.subplots(figsize=(30, 3))

    # Plot absolute error with corresponding colors
    for i, (idx, row) in enumerate(df_accuracy.iterrows()):
        color = colors[i % len(colors)]  # Match the color of actual or forecasted value
        ax_err.bar(idx, row['abs_error'], color=color, label='Absolute Error' if i == 0 else None)

    # Set x-axis ticks to ensure all x-values are listed
    ax_err.set_xticks(df_accuracy.index)
    ax_err.set_xticklabels(df_accuracy.index)
    
    # Title with R² and average absolute error
    plt.title(f"Absolute Error ({col_name}, {type})\nR²: {r2:.2f}, Mean Abs Error: {average_abs_error:.2f}")

    # Rotate x-axis labels for better readability and increase font size
    plt.xticks(rotation=90, fontsize=9)

    # Add legend
    plt.legend()

    plt.show()
    

# CLR transformation function
def clr_transformation(df):
    geometric_mean = np.exp(np.log(df).mean(axis=1))
    clr_df = np.log(df.div(geometric_mean, axis=0))
    return clr_df

# Inverse CLR transformation function
def inverse_clr_transformation(clr_df):
    exp_df = np.exp(clr_df)
    sum_exp = exp_df.sum(axis=1)
    original_df = exp_df.div(sum_exp, axis=0)
    return original_df

# Walk-forward forecasting with AutoReg (DCR)
def walk_forward_forecasting_dcr(deseasonalized_df, periods_df, lags=12):
    predictions = []
    date_lst = []

    for _, row in tqdm(periods_df.iterrows(), total=len(periods_df), desc="Walk-Forward Forecasting (DCR)", dynamic_ncols=True):
        train_data = deseasonalized_df[deseasonalized_df.index <= row["window_end"]]

        if train_data.shape[0] < lags:  # Ensure sufficient data points
            continue

        # Forecast the next period for each column separately
        forecast = []
        for col in train_data.columns:
            model = AutoReg(train_data[col], lags=lags).fit()
            pred = model.predict(start=len(train_data), end=len(train_data))[0]
            forecast.append(pred)

        predictions.append(forecast)
        date_lst.append(row["test"])

    forecast_df = pd.DataFrame(predictions, index=date_lst, columns=deseasonalized_df.columns)
    return forecast_df

# Walk-forward forecasting with VARMAX (TVR)
def walk_forward_forecasting_tvr(deseasonalized_df, periods_df):
    predictions = []
    date_lst = []

    for _, row in tqdm(periods_df.iterrows(), total=len(periods_df), desc="Walk-Forward Forecasting (TVR)", dynamic_ncols=True):
        train_data = deseasonalized_df[deseasonalized_df.index <= row["window_end"]]

        if train_data.shape[0] < 20:  # Ensure sufficient data points
            continue

        # Fit the time-varying regression model
        model = VARMAX(train_data, order=(1, 0))
        model_fit = model.fit(disp=False)

        # Forecast the next period (one month ahead)
        forecast = model_fit.forecast(steps=1)
        predictions.append(forecast.iloc[0].values)
        date_lst.append(row["test"])

    forecast_df = pd.DataFrame(predictions, index=date_lst, columns=deseasonalized_df.columns)
    return forecast_df

# Function to evaluate forecasts
def evaluate_forecasts(actual_df, forecast_df):
    r2_scores = {}
    for col in actual_df.columns:
        r2_scores[col] = r2_score(actual_df[col], forecast_df[col])
    return r2_scores

# Function to perform hypothesis testing
def hypothesis_testing(actual_df, forecast_df):
    p_values = {}
    for col in actual_df.columns:
        _, p_value = ttest_ind(actual_df[col], forecast_df[col])
        p_values[col] = p_value
    return p_values

def cap_extreme_values(df, factor=1.5):
    """ Cap extreme values (via iqr) by replacing with the previous value """
    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # Replace respective outliers with NaN
    df_no_outliers = df.apply(lambda x: np.where((x < lower_bound[x.name]) | (x > upper_bound[x.name]), np.nan, x), axis=0)
    #fillna with previous value
    df_no_outliers.fillna(method='ffill', inplace=True)
    df_no_outliers = df_no_outliers.ffill().bfill()
        
    return df_no_outliers

def deseasonalize_series(series, freq='M'):
    x13_path = '/home/wheelfredie/scripts/BoT_Exports/data/others/x13as'
    series.index = pd.to_datetime(series.index)
    series.index = series.index.to_period(freq)
    result = x13_arima_analysis(series, outlier=False, x12path=x13_path)
    
    # Extract components
    deseasonalized_series = result.seasadj
    trend = result.trend
    irregular = result.irregular
    
    # Calculate seasonal component as the difference
    seasonal = series - trend - irregular
    
    # Seasonal factor is the ratio of the original series to the deseasonalized series
    seasonal_factors = series / deseasonalized_series
    
    return deseasonalized_series, trend, seasonal, irregular, seasonal_factors

# Clean data function
def clean_series(series):
    # Replace inf values with NaN
    series = series.replace([np.inf, -np.inf], np.nan)
    # Drop NaNs
    series = series.dropna()
    return series

# Exponential Smoothing function
def exponential_smoothing(series, alpha=0.3):
    #shift as we cannot see current upcoming IRL
    series = series.shift(1).dropna()
    result = [series[0]]  
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return pd.Series(result, index=series.index)

# Lag-1 model
def lag_1_model(series):
    return series.shift(1).dropna()

#rebuild Series
def rebuild_time_series(start_value, pct_changes, idx):
    """
    Rebuilds the original time series from the start value and percentage changes.

    Parameters:
    start_value (float): The initial value of the time series.
    pct_changes (list of float): The list of percentage changes (expressed as decimals, e.g., 0.05 for 5%).

    Returns:
    pd.Series: The reconstructed time series.
    """
    # Initialize the actual values list with the start value
    actual_values = [start_value]

    # Iteratively calculate the actual values
    for pct_change in pct_changes:
        new_value = actual_values[-1] * (1 + pct_change)
        actual_values.append(new_value)

    # Convert to a pandas Series for easier handling
    actual_series = pd.Series(actual_values, index=idx)
    
    return actual_series

def process_forecast_accuracy(name, plot_show=True, col_name=""):
    # Load the accuracy dataframe
    df_accuracy = pd.read_pickle(f"data/cleaned/SARIMA_RollWalkForward/combined/updated/accuracy/df_accuracy_{name}.pickle")[["actual", "forecast"]]

    # Load the deseasonalized data
    with open(f'data/cleaned/deseasonalised_x13/update/dict_deseasonalized_value_{name}.pickle', 'rb') as handle:
        df_deseasonalized = pickle.load(handle)[name]

    # Get the forecast date
    forecast_date = df_accuracy.index[-1]

    # Reindex the deseasonalized dataframe to match the accuracy dataframe
    df_deseasonalized = df_deseasonalized.reindex(df_accuracy.index)

    # Extract the seasonal factor
    df_seasonal_factor = df_deseasonalized[["seasonal_factor"]]

    # Calculate the last seasonal factor if needed
    last_seasonal_factor = df_seasonal_factor["seasonal_factor"].dropna().iloc[-1]
    df_seasonal_factor.loc[forecast_date] = last_seasonal_factor

    # Load the actual values
    df_actual = pd.read_pickle("data/cleaned/total_export_FirstAnalysis.pkl")[[name]]

    # Create a new index including the previous month
    previous_index = df_accuracy.index[0] - pd.DateOffset(months=1)
    new_index = df_accuracy.index.insert(0, previous_index)

    # Reindex df_actual and insert NaN for the forecast date
    df_actual.loc[forecast_date] = np.nan
    df_actual = df_actual.reindex(new_index)

    # Convert to actual scale by reseasonalizing forecast & actual
    df_accuracy["actual"] = rebuild_time_series(start_value=df_deseasonalized["seasadj"].iloc[0],
                                                pct_changes=df_accuracy["actual"][1:],
                                                idx=df_seasonal_factor.index)
    df_accuracy["forecast"] = (df_accuracy["forecast"].multiply(df_deseasonalized["seasadj"].shift(1))).add(df_deseasonalized["seasadj"].shift(1))

    # Reseasonalize forecast & actual values
    df_accuracy["forecast"] = df_accuracy["forecast"] * df_seasonal_factor["seasonal_factor"]
    df_accuracy["actual"] = df_accuracy["actual"] * df_seasonal_factor["seasonal_factor"]

    # Calculate accuracy metrics
    df_accuracy = calculate_accuracy_metrics(actual=df_accuracy["actual"], forecast=df_accuracy["forecast"]).iloc[1:]
    df_accuracy.to_pickle(f"data/results/binary/{name}.pkl")
    df_accuracy.to_csv(f"data/results/csv/{name}.csv")
    
    # Apply exponential smoothing to the actual data
    df_benchmark = exponential_smoothing(df_accuracy["actual"])
    df_benchmark = calculate_accuracy_metrics(actual=df_accuracy["actual"], forecast=df_benchmark).ffill().bfill()
    df_benchmark.to_pickle(f"data/results/binary/Benchmark_{name}.pkl")
    df_benchmark.to_csv(f"data/results/csv/Benchmark_{name}.csv")


    # Plot actual vs forecast
    if plot_show:
        plot_actual_vs_forecast(df_accuracy, col_name=col_name)
        plot_actual_vs_forecast(df_benchmark, col_name="Benchmark-Exponential Smoothing (Export-BOP)")

    return df_accuracy, df_benchmark

def convert_period_index_to_datetime(df):
    """
    Convert the PeriodDtype index of a DataFrame to timestamps and then to datetime.

    Parameters:
    df (pd.DataFrame): The DataFrame with a PeriodDtype index.

    Returns:
    pd.DataFrame: The DataFrame with the index converted to datetime.
    """
    if isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.to_timestamp()
    df.index = pd.to_datetime(df.index)
    return df

def calculate_weighted_forecast(name, mapper_path, weights_path, plot_show=True,
                                col_name="Level 1 (Custom Basis)"):
    # Load mapper and extract constituents
    mapper = pd.read_pickle(mapper_path)
    map_l1 = mapper.iloc[:, [-1]].dropna()
    name = map_l1.columns[0]
    constituents = map_l1[name].to_list()

    # Load weights
    with open(weights_path, "rb") as f:
        weights_r2 = pickle.load(f)[name]

    max_row_names = weights_r2.idxmax()
    weight_technique_name_map = {"Lag-1": "Lag_1_", "Exp Smooth": "Exp_Smooth_", "DCR": "DCR_"}
    max_row_names = max_row_names.map(weight_technique_name_map)

    df_weighted = pd.DataFrame()

    for i in constituents:
        best_r2 = max_row_names[i]
        w = convert_period_index_to_datetime(pd.read_pickle(f"data/cleaned/forecasted_weights/{best_r2}{name}.pkl")[i])
        forecast, _ = process_forecast_accuracy(name=i, plot_show=False)
        forecast = forecast["forecast"]

        if len(w) > len(forecast):
            w = w.reindex(forecast.index)
        else:
            forecast = forecast.reindex(w.index)

        df_weighted[i] = forecast * w

    df_weighted = df_weighted.dropna().sum(axis=1).dropna()

    df_actual, _ = process_forecast_accuracy(name=name, plot_show=False)
    df_actual = df_actual["actual"]


    df_actual = df_actual.reindex(df_weighted.index)

    df_accuracy = calculate_accuracy_metrics(actual=df_actual, forecast=df_weighted)
    if plot_show:
        plot_actual_vs_forecast(df_accuracy=df_accuracy, col_name=col_name )

    return df_accuracy

def calculate_weighted_forecast_with_coverage_adjustment(name, df_weighted, mapper_path, 
                                                         weights_path, plot_show=True,
                                                         col_name="Level 1 (BOP inclusive Coverage Adjustment)"):
    df_Coverage_adj, _ = process_forecast_accuracy(name="Coverage_Adjustment", plot_show=False)
    df_Coverage_adj = df_Coverage_adj["forecast"]

 
    df_Coverage_adj = df_Coverage_adj.reindex(df_weighted.index)

    df_weighted = df_weighted + df_Coverage_adj

    df_actual, _ = process_forecast_accuracy(name=name, plot_show=False)
    df_actual = df_actual["actual"]


    df_actual = df_actual.reindex(df_weighted.index)

    df_accuracy = calculate_accuracy_metrics(actual=df_actual, forecast=df_weighted)
    if plot_show:
        plot_actual_vs_forecast(df_accuracy=df_accuracy, col_name=col_name)
        
    return df_accuracy
    
    
def optimize_forecast(df_accuracy, plot_show=True, col_name="Level 1 (Custom Basis) WITH LINEAR COMBINATION OPTIMISATION"):
    # Copy the DataFrame to avoid modifying the original
    df = df_accuracy.copy(deep=True)

    # Define the objective function to minimize
    def objective(params, actual, forecast):
        a, b = params
        adjusted_forecast = a * forecast + b
        error = np.mean(np.abs(actual - adjusted_forecast))  # Mean Absolute Error (MAE)
        return error

    # Initial guess for a and b
    initial_params = [1, 0]

    # Perform the optimization
    result = minimize(objective, initial_params, args=(df['actual'], df['forecast']))

    # Extract the optimal parameters
    a_opt, b_opt = result.x

    # Apply the optimal parameters to adjust the forecast
    df['forecast'] = a_opt * df['forecast'] + b_opt
    
    df = calculate_accuracy_metrics(actual=df['actual'], forecast=df['forecast'])

    # Plot the results
    if plot_show:
        plot_actual_vs_forecast(df_accuracy=df, col_name=col_name)
    
    return df