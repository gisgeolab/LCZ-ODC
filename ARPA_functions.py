from sodapy import Socrata
import pandas as pd
from datetime import datetime, timedelta
import requests
from io import BytesIO
from zipfile import ZipFile
import os
import time
import numpy as np
import dask.dataframe as dd

"---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

def order_dates(df):
    """
    Orders the dates of a dataframe
    
    Parameters
        df: dataframe containing datetime column
    
    """
    df = df.sort_values(by='data', ascending=True).reset_index(drop=True)
    return df

"---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

def connect_ARPA_api(token: str) -> Socrata:
    """
    Function to connect to ARPA API. Unauthenticated client only works with public data sets, and there is a limit for the requests.
    Note 'None' in place of application token, and no username or password.
    To get all the available data from the API the authentication is required.
    Parameters:
        token (str): the ARPA token obtained from Open Data Lombardia website
    Returns:
        client: client session
    """
    # Connect to Open Data Lombardia using the token
    if token == "":
        print("No token provided. Requests made without an app_token will be subject to strict throttling limits.")
        client = Socrata("www.dati.lombardia.it", None)
    else:
        print("Using provided token.")
        client = Socrata("www.dati.lombardia.it", app_token=token)

    return client

"---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

def ARPA_sensors_info(client) -> pd.DataFrame:
    """
    Functions to convert sensors information to Pandas dataframe and fix the data types.
    Parameters:
        sensors_info: object obtained from Socrata with get request
    Returns:
        df: dataframe containing ARPA sensors information
    """
    
    stationsId = "nf78-nj6b" # Select meteo stations dataset containing positions and information about sensors
    sensors_info = client.get_all(stationsId)

    sensors_df = pd.DataFrame(sensors_info)
    sensors_df["idsensore"] = sensors_df["idsensore"].astype("int32")
    sensors_df["tipologia"] = sensors_df["tipologia"].astype("category")
    sensors_df["idstazione"] = sensors_df["idstazione"].astype("int32")
    sensors_df["quota"] = sensors_df["quota"].astype("int16")
    sensors_df["provincia"] = sensors_df["provincia"].astype("category")
    sensors_df["storico"] = sensors_df["storico"].astype("category")
    sensors_df["datastart"] = pd.to_datetime(sensors_df["datastart"])
    sensors_df["datastop"] = pd.to_datetime(sensors_df["datastop"])
    sensors_df = sensors_df.drop(columns=[":@computed_region_6hky_swhk", ":@computed_region_ttgh_9sm5"])

    return sensors_df

"---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

def check_dates(start_datetime, end_datetime):
    """
    Check that the start and end dates are in the same year.
    
    Parameters:
        start_date (datetime): The start date in the format "YYYY-MM-DD".
        end_date (datetime): The end date in the format "YYYY-MM-DD".
    
    Returns:
        year (int): The year of the start and end dates.
        start_datetime (datetime): The start date as a datetime object.
        end_datetime (datetime): The end date as a datetime object.
        
    Raises:
        Exception: If the start and end dates are not in the same year.
    """

    # Check that the start and end dates are in the same year
    if start_datetime.year != end_datetime.year:
        raise Exception("Dates must be in the same year! Years chosen are: {year_start} and {year_end}".format(
            year_start=start_datetime.year, year_end=end_datetime.year
        ))
    elif start_datetime > end_datetime:
        raise Exception("Start date bust be before end date")

    # Get the year of the start and end dates
    year = start_datetime.year
    
    return year, start_datetime, end_datetime

"---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

def req_ARPA_start_end_date_API(client):
    """
    Function to request the start and the end date of data available in the ARPA API.
        Parameters:
            client: the client session
        Returns: 
            start_API_date (str): starting date for available data inside the API.
            end_API_date (str): ending date for available data inside the API.
        
    """
    
    #Weather sensors dataset id on Open Data Lombardia
    weather_sensor_id = "647i-nhxk"
    
    #Query min and max dates
    query = """ select MAX(data), MIN(data) limit 9999999999999999"""

    #Get max and min dates from the list
    min_max_dates = client.get(weather_sensor_id, query=query)[0]
    
    #Start and minimum dates from the dict obtained from the API
    start_API_date = min_max_dates['MIN_data']
    end_API_date = min_max_dates['MAX_data']
    
    #Convert to datetime
    start_API_date = datetime.strptime(start_API_date, "%Y-%m-%dT%H:%M:%S.%f")
    end_API_date = datetime.strptime(end_API_date, "%Y-%m-%dT%H:%M:%S.%f")
    
    print("The data from the API are available from: " ,start_API_date, " up to: ", end_API_date, ". Select data in this range if you want to use API data.")

    return start_API_date, end_API_date

"---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

def req_ARPA_data_API(client, start_date, end_date, sensors_list):
    """
    Function to request data from available weather sensors in the ARPA API using a query.
        Parameters:
            client: the client session
            start date (str): the start date in yyy-mm-dd format
            end date (str): the end date in yyy-mm-dd format
        Returns: 
            time_series: time series of values requested with the query for all sensors
        
    """
    
    #Select the Open Data Lombardia Meteo sensors dataset
    weather_sensor_id = "647i-nhxk"
    
    #Convert to string in year-month-day format, accepted by ARPA query
    start_date = start_date.strftime("%Y-%m-%dT%H:%M:%S.%f")
    end_date = end_date.strftime("%Y-%m-%dT%H:%M:%S.%f")
    
    print("--- Starting request to ARPA API ---")
    
    t = time.time()
    
    #Query data
    query = """
      select
          *
      where data >= \'{}\' and data <= \'{}\' limit 9999999999999999
      """.format(start_date, end_date)

    #Get time series and evaluate time spent to request them
    time_series = client.get(weather_sensor_id, query=query)
    
    print(time_series)
    elapsed = time.time() - t
    print("Time used for requesting the data from ARPA API: ", elapsed)
    
    #Create dataframe
    df = pd.DataFrame(time_series, columns=['idsensore','data','valore','stato'])
    df = df[df.stato.isin(["VA", "VV"])]  #keep only validated data identified by stato equal to VA and VV
    df = df.drop(['stato'], axis=1)
    
    #Convert types
    df['valore'] = df['valore'].astype('float32')
    df['idsensore'] = df['idsensore'].astype('int32')
    df['data'] = pd.to_datetime(df['data'])
    df = df.sort_values(by='data', ascending=True).reset_index(drop=True)
    
    #Filter with selected sensors list
    try:
        df = df[df['value'] != -9999]
    except:
        df = df[df['valore'] != -9999]
    df = df[df['idsensore'].isin(sensors_list)]

    return df

"---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

def download_extract_csv_from_year(year):
    """
    Function for selecting the correct link for downloading zipped .csv meteorological data from ARPA sensors and extracting it.
    For older data it is necessary to download this .csv files containing the time series of the meteorological sensors.
        Parameters:
            year(str): the selected year for downloading the .csv file containing the meteorological sensors time series
        Returns:
            None
    """
    
    #Create a dict with years and link to the zip folder on Open Data Lombardia - REQUIRES TO BE UPDATED EVERY YEAR
    switcher = {
        '2023': "https://www.dati.lombardia.it/download/48xr-g9b9/application%2Fzip",
        '2022': "https://www.dati.lombardia.it/download/mvvc-nmzv/application%2Fzip",
        '2021': "https://www.dati.lombardia.it/download/49n9-866s/application%2Fzip",
        '2020': "https://www.dati.lombardia.it/download/erjn-istm/application%2Fzip",
        '2019': "https://www.dati.lombardia.it/download/wrhf-6ztd/application%2Fzip",
        '2018': "https://www.dati.lombardia.it/download/sfbe-yqe8/application%2Fzip",
        '2017': "https://www.dati.lombardia.it/download/vx6g-atiu/application%2Fzip",
        '2016': "https://www.dati.lombardia.it/download/kgxu-frcw/application%2Fzip",
        '2015': "https://www.dati.lombardia.it/download/knr4-9ujq/application%2Fzip",
        '2014': "https://www.dati.lombardia.it/download/fn7i-6whe/application%2Fzip",
        '2013': "https://www.dati.lombardia.it/download/76wm-spny/application%2Fzip",
    }
    
    #Select the url and make request
    url = switcher[year]
    filename = 'meteo_'+str(year)+'.zip'
    
    #If yrar.csv file is already downloaded, skip download
    if not os.path.exists(year+".csv"):
        print("--- Starting download ---")
        t = time.time()
        print(('Downloading {filename} -> Started. It might take a while... Please wait!').format(filename = filename))
        response = requests.get(url, stream=True)
        
        block_size = 1024
        wrote = 0 
        
        # Writing the file to the local file system
        with open(filename, "wb") as f:
            for data in response.iter_content(block_size):
                wrote = wrote + len(data)
                f.write(data)
                percentage = wrote / (block_size*block_size)
                print("\rDownloaded: {:0.2f} MB".format(percentage), end="")
            
        elapsed = time.time() - t
        
        print(('\nDownloading {filename} -> Completed. Time required for download: {time:0.2f} s.').format(filename = filename, time=elapsed))

        print(("Starting unzipping: {filename}").format(filename=filename))
        #Loading the .zip and creating a zip object
        with ZipFile(filename, 'r') as zObject:
            # Extracting all the members of the zip into a specific location
            zObject.extractall()

        csv_file=str(year)+'.csv'
        print(("File unzipped: {filename}").format(filename=filename))
        print(("File csv saved: {filename}").format(filename=csv_file))

        #Remove the zip folder
        if os.path.exists(filename):
            print(("{filename} removed").format(filename=filename))
            os.remove(filename)
        else:
            print(("The file {filename} does not exist in this folder").format(filename=filename))
    
    else:
        print(year+".csv already exists. It won't be downloaded.")
        
"---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
        
def process_ARPA_csv(csv_file, start_date, end_date, sensors_list):
    """
    This function reads the ARPA csv file into a dask dataframe and provided a computed dataframe. It renames the columns like the API columns names, filters between provided dates and select the sensors present in the list.
        Parameters:
            csv_file(str): name of the csv file
            start_date(datetime): start date for processing
            end_date(datetime): end date for processing
            sensors_list(string list): list of selected sensors
        Returns:
            df(dataframe): computed filtered dask dataframe
    """
    
    print("--- Starting processing csv data ---")
    print(("The time range used for the processing is {start_date} to {end_date}").format(start_date=start_date,end_date=end_date))
    
    #Read csv file with Dask dataframe
    df = dd.read_csv(csv_file, usecols=['IdSensore','Data','Valore', 'Stato']) 
    
    #Make csv columns names equal to API columns names
    df = df.rename(columns={'IdSensore': 'idsensore', 'Data': 'data', 'Valore': 'valore', 'Stato':'stato'})
    
    #Type formatting
    df['valore'] = df['valore'].astype('float32')
    df['idsensore'] = df['idsensore'].astype('int32')
    df['data'] = dd.to_datetime(df.data, format='%d/%m/%Y %H:%M:%S')
    df['stato'] = df['stato'].astype(str)
    
    #Filter using the dates
    df = df[df['valore'] != -9999]
    df = df.loc[(df['data'] >= start_date) & (df['data'] <= end_date)]
    #Filter on temperature sensors list
    sensors_list = list(map(int, sensors_list))
    df = df[df['idsensore'].isin(sensors_list)] #keep only sensors in the list (for example providing a list of temperature sensors, will keep only those)
    df = df[df.stato.isin(["VA", "VV"])] #keep only validated data identified by stato equal to VA and VV
    df = df.drop(['stato'], axis=1)

    #Order dates
#    df = df.sort_values(by='data', ascending=True).reset_index(drop=True)
    
    print("Starting computing dataframe")
    
    #Compute df
    t = time.time()
    df = df.compute()
    elapsed = time.time() - t
    print("Time used for computing dataframe {time:0.2f} s.".format(time=elapsed))
    
    return df 

"---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

def outlier_filter_iqr(df):
    """
    Function for filtering using quantiles
    
        Parameters:
            df (dataframe): ARPA dataframe containing at least the following columns: "idsensore"(int), "data"(datetime) and "valore"(float)
            sensors_list (int list): list of sensors
        Returns:
            df(dataframe): filtered dataframe using IQR
    
    """
    
    Q1 = df['Value'].quantile(0.25)
    Q3 = df['Value'].quantile(0.75)
    IQR = Q3 - Q1
    mask = (df['Value'] < (Q1 - 1.5 * IQR)) | (df['Value'] > (Q3 + 1.5 * IQR))
    
    df.loc[mask, 'Value'] = np.nan
    return df

"---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

def outlier_filter_zscore(df, sensors_list, threshold=3):
    """
    Filter dataframe using Z-Score method
        Parameters:
            df (pandas.DataFrame): ARPA dataframe with columns "idsensore" (int), "data" (datetime), and "valore" (float)
            sensors_list (list of ints): list of sensor ids
            threshold (float, optional): Z-Score threshold to use for filtering, default is 3
        Returns:
            pandas.DataFrame: filtered dataframe using Z-Score method
        
    """
    filtered_df = df.copy()

    for sensor in sensors_list:
        sensor_df = filtered_df[filtered_df['Sensor ID'] == sensor]
        mean = sensor_df['Value'].mean()
        std = sensor_df['Value'].std()
        z = (sensor_df['Value'] - mean) / std
        filtered_df.loc[(filtered_df['Sensor ID'] == sensor) & (z.abs() > threshold), 'Value'] = np.nan
        
    return filtered_df

"---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

def aggregate_group_data(df, temporal_agg="D"):
    """
    Aggregates ARPA data with providing a temporal aggregation (day, week etc.) and a statistical aggregration function (mean, max, min etc.). The dataframe is grouped by sensor id (idsensore).
            Parameters:
                df(dataframe): ARPA dataframe containing the following columns: "idsensore"(int), "data"(datetime) and "valore"(float)
                temporal_agg(str): the temporal aggregation accepted by the resample() method (D, M, Y or others)
            Returns:
                df(dataframe): computed filtered and aggregated dask dataframe
    """
    
    print("Number of sensors available in the dataframe: ", len(df['Sensor ID'].unique()))
    print("Temporal aggregation: " + temporal_agg)
    df = df.set_index('DateTime')
          
    grouped = df.groupby('Sensor ID').resample(str(temporal_agg))['Value'].agg(['count', 'min', 'max', 'mean', 'median', 'std']).round(2)
    grouped = grouped.reset_index()
    
    return grouped

"---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"

def aggregate_group_data_wind_dir(df, temporal_agg="D"):
    """
    Aggregates Wind Direction ARPA data with providing a temporal aggregation (day, week etc.) and a statistical aggregration function (mode, count). The dataframe is grouped by sensor id (idsensore).
            Parameters:
                df(dataframe): ARPA dataframe containing the following columns: "idsensore"(int), "data"(datetime) and "valore"(float)
                temporal_agg(str): the temporal aggregation accepted by the resample() method (D, M, Y or others)
            Returns:
                df(dataframe): computed filtered and aggregated dask dataframe
    """
    
    print("Number of sensors available in the dataframe: ", len(df.idsensore.unique()))
    print("Temporal aggregation: " + temporal_agg)
    df = df.set_index('data')
          
    grouped = df.groupby('idsensore').resample(str(temporal_agg))['valore'].agg([lambda x: pd.Series.mode(x)[0], 'count']).rename({'<lambda_0>': 'mode'}, axis=1)
    grouped = grouped.reset_index()
    
    return grouped