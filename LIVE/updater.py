#%%
from helper import *

import sys
import os
import shutil
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, UnexpectedAlertPresentException


def archive_old_downloads(download_dir, archive_dir):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for filename in os.listdir(download_dir):
        file_path = os.path.join(download_dir, filename)
        if os.path.isfile(file_path):
            name, ext = os.path.splitext(filename)
            new_archive_path = os.path.join(archive_dir, f"{name}_{timestamp}{ext}")
            shutil.move(file_path, new_archive_path)

#%%
df_export_ANALYSIS = pd.read_pickle("data/cleaned/total_export_FirstAnalysis.pkl")
target = df_export_ANALYSIS.columns.to_list()
target.remove('Timing_Adjustment')

display(df_export_ANALYSIS)


#%%
#Load Historical Forecast until last available month

df_export_ANALYSIS = pd.read_pickle("data/clean/total_export.pkl")

display(df_export_ANALYSIS)
display(df_export_ANALYSIS.columns)

#%%
current_date = pd.Timestamp.today().date()
latest_month_historically_available_ACTUAL = df_export_ANALYSIS.index.max()
#forecast is one period ahead
latest_month_historically_available_FORECAST = latest_month_historically_available_ACTUAL + pd.DateOffset(months=1)
display(f"Historically Avalilable Actual Data (since last run): {latest_month_historically_available_ACTUAL}")
display(f"Historically Avalilable Forecasted Data (since last run): {latest_month_historically_available_FORECAST}")
#convert actual data month to "MONTH"
#get a list of months from latest_month_historically_available_ACTUAL to current_date
months_list = pd.date_range(start=latest_month_historically_available_ACTUAL, end=current_date, freq='MS')[1:]
months = months_list.strftime('%B %Y').to_list()
#convert months to like str "janurary"
month_start = list(map(lambda x: x.split(" ")[0], months))[0]
months_end = list(map(lambda x: x.split(" ")[0], months))[0:][::-1] #list because i will interate backwards in case latest month is unavailable
year_start = list(map(lambda x: x.split(" ")[1], months))[0]
years_end = list(map(lambda x: x.split(" ")[1], months))[0:][::-1]
# display(months_start, months_end, year_start, year_end)

data_downloaded = False

#download directory
#get current path
current_path = os.getcwd()
#downlaod directory is current path + data/BoT_raw_download/
download_dir = os.path.join(current_path, "data/BoT_raw_download")
archive_dir = os.path.join(download_dir, "archive")

#Ensure the directory exists
os.makedirs(download_dir, exist_ok=True)
os.makedirs(archive_dir, exist_ok=True)

#Archive existing files before new download
archive_old_downloads(download_dir, archive_dir)

#Set up Chrome options to configure download directory
chrome_options = webdriver.ChromeOptions()
prefs = {
    "download.default_directory": download_dir,
    "download.prompt_for_download": False,
    "directory_upgrade": True,
    "safebrowsing.enabled": True
}
chrome_options.add_experimental_option("prefs", prefs)

for month_end in tqdm(months_end,
                        desc="Attempting Download of Actual Data From BoT",
                        dynamic_ncols=True):
    
    year_end = years_end[months_end.index(month_end)]
    display(f"Attempting to download actual data from BoT Website for: {month_end} {year_end}")
    
    
    
    try:
        
        # Set up the WebDriver
        driver = webdriver.Chrome(options=chrome_options)  # or webdriver.Firefox() for Firefox
        driver.get("https://app.bot.or.th/BTWS_STAT/statistics/BOTWEBSTAT.aspx?reportID=979&language=ENG")

        # Select Frequency
        frequency_select = Select(driver.find_element(By.ID, 'drpPeriod'))  
        frequency_select.select_by_value('MTH')

        # Select Start Month
        start_month_select = Select(driver.find_element(By.ID, 'drpFromMonth'))  
        start_month_select.select_by_visible_text(month_start)
        
        # Select Start Year
        start_year_select = Select(driver.find_element(By.ID, 'drpFromYear'))  
        start_year_select.select_by_visible_text(year_start)

        # Select End Month
        end_month_select = Select(driver.find_element(By.ID, 'drpToMonth'))  
        end_month_select.select_by_visible_text(month_end)
        
        # Select End Year
        end_year_select = Select(driver.find_element(By.ID, 'drpToYear'))  
        end_year_select.select_by_visible_text(year_end)

        # Submit the form
        driver.find_element(By.ID, 'btnSubmit').click()  
        
        try:
            # Wait for alert to appear (means invalid period selected)
            #if alert appears, accept and move to the next date
            WebDriverWait(driver, 5).until(EC.alert_is_present())
            alert = driver.switch_to.alert
            alert.accept()
            print(f"Selected period is invalid for {month_end} {year_end}. Moving to the next date.")
            driver.quit()
            #pop it from available months
            months_list = months_list[:-1]
            continue
        
        except TimeoutException:
            pass  

        #otherwise if no alert, download the data (data period is valid)
        #Download the CSV
        download_button = driver.find_element(By.ID, 'imbExportTable')
        download_button.click()

        # Wait for the download to complete
        time.sleep(10)
        data_downloaded = True
        driver.quit()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    if data_downloaded:
        display(f"Actual Data Downloaded Successfully for BoT. Latest Data Available: {month_end} {year_end}")
        break
if data_downloaded == False:
    display(f"No data available for download from BoT. Latest Data Available: {latest_month_historically_available_ACTUAL}")
    sys.exit(1)


#%%
#read the excel file form the files downloaded
filename = os.listdir(download_dir)[0]
df_new_export = pd.read_excel(os.path.join(download_dir, filename))

#get headers
COLNAME_df_new_export = df_new_export.iloc[4].to_list()
COLNAME_df_new_export[1] = "class"
COLNAME_df_new_export = COLNAME_df_new_export[1:]

#remove unwanted rows 
df_new_export = df_new_export.iloc[5:101,1:]
df_new_export.columns = COLNAME_df_new_export
df_new_export.reset_index(drop=True, inplace=True)

#remove "/" from class names and standardize the class names
df_new_export["class"] = df_new_export["class"].str.replace("/", "")
#standardise names
df_new_export["class"] = [col.strip().replace(" ", "_") if isinstance(col, str) and col.strip() != "" else col for col in df_new_export["class"]]
df_new_export.set_index("class", inplace=True, drop=True)

#fix the dates
dates = df_new_export.columns
fixed_dates = []
wrong_dates = []
for date in dates:
    try:
        pd.to_datetime(date, format='%b %Y ')
        fixed_dates.append(date)
    except ValueError as e:
        date = date.replace('r', '')
        fixed_dates.append(date)
        
for date in fixed_dates:
    try:
        pd.to_datetime(date, format='%b %Y ')
    except ValueError as e:
        wrong_dates.append(date)

#if any one month-dates have issues, kill the process
if len(wrong_dates) > 0:
    display(f"Error in the following dates: {wrong_dates}")
    display(f"Please fix the dates before proceeding")
    sys.exit(1)
    
df_new_export.columns = pd.to_datetime(fixed_dates, format='%b %Y ')
df_new_export = df_new_export.T
df_new_export = df_new_export[df_export_ANALYSIS.columns]
df_new_export.sort_index(ascending=True, inplace=True)

df_export_ANALYSIS = pd.concat([df_export_ANALYSIS, df_new_export], axis=0)
df_export_ANALYSIS.sort_index(ascending=True, inplace=True)
display(df_export_ANALYSIS)


#%%
# target = df_export_ANALYSIS.columns.to_list()
# display("len:", len(target))
# display(f"target: {target}")

# for name in tqdm(target,
#                     desc="Processing",
#                     dynamic_ncols=True):
display("df_export_ANALYSIS")
display(set(map(lambda x : type(x), df_export_ANALYSIS.index)))

name = 'Exports,_f.o.b._(BOP_basis)' 
    
print(f"Processing {name}...")
col= name
cols = [name]
df = df_export_ANALYSIS[[name]]
if name in ['Crude_oil', 'Photographic_&_cinematographic_instruments_&_supplies']:
    #DATA IMPERFECTIONS
    df = df[df.index >= "1996-11-01"]
df[name] = df[name].replace(0, np.nan).ffill().bfill()

ts = df[name].dropna()

# X13-ARIMA-SEATS deseasonalization
#DEsesonality using X13 by US Census Bureau

# Suppress specific X13 warnings
warnings.filterwarnings("ignore", category=X13Warning)

x13_path='/home/wheelfredie/scripts/BoT_Exports/data/others/x13as'
freq='M'

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
#pickle
df1.to_pickle(f'data/clean/deseasonalised/{name}.pickle')
display("df1")
display(set(map(lambda x : type(x), df1.index)))
display(df1)


adj_ts = df1['seasadj'].pct_change().dropna()
    
#Walk Forward parameters
window_size = 180  # 15 years of monthly data
total_observations = len(adj_ts)  # Total data points
steps = total_observations - window_size  # Remaining points after initial training window

# Generate periods DataFrame
periods_df = generate_periods_df(adj_ts, window_size, steps)

#convert months list to period
months_list = pd.to_datetime(months_list).to_period('M')
periods_df = periods_df[periods_df["window_end"].isin(months_list)]
for col in periods_df.columns:
        periods_df[col] = periods_df[col].dt.to_timestamp()

# Perform walk-forward forecasting
rolling_forecasts = walk_forward_forecasting(adj_ts, periods_df)

forecast = pd.DataFrame(rolling_forecasts, columns=['forecast'])
adj_ts.index = adj_ts.index.to_timestamp()
actual = adj_ts[adj_ts.index.isin(forecast.index)]

df_accuracy = calculate_accuracy_metrics(actual, forecast)

#get previous data
with open(f'data/clean/SARIMA_RollWalkForward/accuracy/df_accuracy_{name}.pickle',
        'rb') as handle:
    df_accuracy_prev = pickle.load(handle)
    
with open(f'data/clean/SARIMA_RollWalkForward/periods/periods_df_{name}.pickle',
        'rb') as handle:
    periods_df_prev = pickle.load(handle)
    
#combine the previous and current data
df_accuracy = pd.concat([df_accuracy_prev, df_accuracy])
periods_df = pd.concat([periods_df_prev, periods_df])

#save the df_accuracy & periods_df as pickle
with open(f'data/clean/SARIMA_RollWalkForward/updated/accuracy/df_accuracy_{name}.pickle',
        'wb') as handle:
    pickle.dump(df_accuracy, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f'data/clean/SARIMA_RollWalkForward/updated/periods/periods_df_{name}.pickle',
        'wb') as handle:
    pickle.dump(periods_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

#open the pickle file
with open(f'data/clean/SARIMA_RollWalkForward/accuracy/df_accuracy_{name}.pickle',
        'rb') as handle:
    df_accuracy_prev = pickle.load(handle)

#convert forecast into dollar value

if name == 'Exports,_f.o.b._(BOP_basis)':
    L0, BM = process_forecast_accuracy(name, plot_show=True, col_name=col)

else:
    #handle weights


#%%
L0, BM = process_forecast_accuracy(name, plot_show=True, col_name=col)
        

# %%
L0
# %%
BM
# %%
