import copy
import dcor
import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from functools import reduce

"""
HPC Consumption Data, and its pre-processing

Note: For now assumpts hourly sampled data
"""
class ConsumptionData:
    def __init__(self):
        self.__dbClient = None
        self.__data = None
        self.__timeCols = ['_time', 'month', 'day', 'dayofweek', 'hour']

    """
    Configure connection to InfluxDB
    """
    def setConnection(self, serverURL, token, org):
        self.__dbClient = InfluxDBClient(url=serverURL, token=token, org=org)

    """
    Params:
        startTime, stopTime, nanThreshold=0.03

        Fetches the raw data from 'startTime' to 'stopTime' and creates an overall table.
        Columns with NaN values in a share of 'nanThreshold' are dropped.

    Resulting columns:
      _time : [datetime], timestamp, aggregation of per-hour data
      carbonIntensity : [kg/MWh] Calculated carbon intensity of measured energy production
      fossilFuelPercentage [%] : Share of fossil energy in the overall carbon intensity
      fossilEnergy    : [MWh] Energy production from Braunkohle, Steinkohle, Erdgas, 'Sonstige Konventionelle'
      renewableEnergy : [MWh] Energy production from Biomasse, 'Wind Onshore', 'Wind Offshore', Photovoltaik, Wasserkraft, 'Sonstige Erneuerbare'
      price           : [€/MWh] monetary energy costs
      Weather data columns, namely:
          cloud_cover, dew_point,  precipitation,  pressure_msl, relative_humidity, solar, sunshine,
          temperature, visibility, wind_speed
      Additional time columns, a breakdown of the _time column:
           month, day, dayofweek, dayofyear, hour

    Note: Energy carriers Kernenergie and Pumpspeicher are ignored.

    Returns:
        Deep copy of the fetched data
    """
    def fetch(self, startTime, stopTime, nanThreshold=0.03):
        query = f'''from(bucket:"price")
            |> range(start: {startTime}, stop: {stopTime})
            |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        price_df = self.__dbClient.query_api().query_data_frame(query)
        price_df = price_df.drop(columns=['result', '_start', '_stop', '_measurement', 'region', 'table'])
        price_df = price_df.rename(columns={'Deutchland/Luxembourg': 'price'})
        price_df['_time'] = pd.to_datetime(price_df['_time'])
        price_df = price_df.set_index('_time')
#        price_df = price_df.sort_index()

        query = f'''from(bucket:"co2")
            |> range(start: {startTime}, stop: {stopTime})
            |> filter(fn: (r) => r._measurement == "co2_calculated")
            |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")'''
        co2_df = self.__dbClient.query_api().query_data_frame(query)
        co2_df = co2_df.drop(columns=['result', '_start', '_stop', '_measurement', 'countryCode', 'table'])
        co2_df['_time'] = pd.to_datetime(co2_df['_time'])
        co2_df = co2_df.set_index('_time')
#        co2_df = co2_df.sort_index()

        query = f'''from(bucket:"energy")
            |> range(start: {startTime}, stop: {stopTime})
            |> filter(fn: (r) => r._field != "Kernenergie" and r._field != "Pumpspeicher")
            |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        energy_df = self.__dbClient.query_api().query_data_frame(query)
        # Only care about distinguishing fossil and renewable energy carriers
        energy_df['fossilEnergy'] = energy_df['Braunkohle'] + energy_df['Steinkohle'] + energy_df['Erdgas'] + energy_df['Sonstige Konventionelle']
        energy_df['renewableEnergy'] = energy_df['Biomasse'] + energy_df['Wind Onshore'] + energy_df['Wind Offshore'] + energy_df['Photovoltaik'] + energy_df['Wasserkraft'] + energy_df['Sonstige Erneuerbare']
        dropCols = [
            'result', '_start', '_stop', '_measurement', 'region', 'table',
            'Braunkohle', 'Steinkohle', 'Erdgas', 'Sonstige Konventionelle',
            'Biomasse', 'Wind Onshore', 'Wind Offshore', 'Photovoltaik', 'Wasserkraft', 'Sonstige Erneuerbare',
        ]
        energy_df = energy_df.drop(columns=dropCols)
        energy_df['_time'] = pd.to_datetime(energy_df['_time'])
        energy_df = energy_df.set_index('_time')
#        energy_df = energy_df.sort_index()

        query = f'''from(bucket:"weather")
            |> range(start: {startTime}, stop: {stopTime})
            |> filter(fn: (r) => r._measurement == "weather" and r._field != "wind_direction" and r._field != "wind_gust_direction")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        '''
        weather_df = self.__dbClient.query_api().query_data_frame(query)
        weather_df = weather_df.drop(columns=['result', '_start', '_stop', '_measurement', 'condition', 'icon', 'source_id', 'table'])

        # unite the tables into one
        dfs = [price_df, co2_df, energy_df, weather_df]
        self.__data = reduce(lambda left, right: pd.merge(left, right, on='_time', how='inner'), dfs)

        # remove columns with too much NaN
        self.__data = self.__data[[c for c in self.__data if self.__data[c].isna().sum() < nanThreshold * len(self.__data)]]

        # add time-related columns
        self.__data['month'] = self.__data['_time'].dt.month
        self.__data['day'] = self.__data['_time'].dt.day
        self.__data['dayofweek'] = self.__data['_time'].dt.dayofweek
        self.__data['dayofyear'] = self.__data['_time'].dt.dayofyear
        self.__data['hour'] = self.__data['_time'].dt.hour

        return self.__data.copy()

    """
    Copy constructor
    """
    def deepCopy(self):
        dc = ConsumptionData()
        dc.set(self.get())
        return dc

    """
    Returns deep copy of the stored data
    """
    def get(self):
        return self.__data.copy()

    """
    Set a deep copy of the provided data
    """
    def set(self, data):
        self.__data = data.copy()

    """
    Write to and read from a CSV file under given path
    """
    def toCSV(self, fpath):
        self.__data.to_csv(fpath, index=False)
    def fromCSV(self, fpath):
        self.__data = pd.read_csv(fpath, parse_dates=["_time"])

    """
    Removes the specified columns, and returns deep copy of the new data set
    """
    def dropCols(self, cols):
        self.__data.drop(columns=cols, inplace=True)
        return self.__data.copy()

    """
    Keeps the specified data columns (as well as the '_time' column), and
    returns deep copy of the new data set
    """
    def keepCols(self, cols):
        self.__data = self.__data.loc[:, pd.Index(['_time'] + cols.to_list())]
        return self.__data.copy()

    """
    Returns number of NaN values per data column
    """
    def nanCount(self):
        return self.__data.isna().sum()

    """
    Drop rows with NaN values
    """
    def dropNAN(self):
        self.__data.dropna(inplace=True)

    """
    Drop given number of data's head rows
    """
    def dropHeadRows(self, number):
        self.__data = self.__data[number:]

    """
    Append the given data rows
    """
    def appendRows(self, rows):
        self.__data = pd.concat([self.__data, rows])

    """
    Returns the shape of the data
    """
    def shape(self):
        return self.__data.shape

    """
    Transforms absolute measurement values into gradient values

    The gradients between 2 rows at times t1 and t2 are calculated by
        gradients(t2) = values(t2) - values(t1)
    There's no division by the time difference as provision of hourly non-NaN data is assumed,
    i.e. it always is t2 - t1 = 1. Might be refined at need.

    Returns:
        Deep copy of the gradient-turned data
    """
    def toGradient(self):
        cols = [c for c in self.__data.columns if c not in self.__timeCols] # get names of data columns
        self.__data.loc[:, cols] = self.__data.loc[:, cols].diff() # get the (hourly) changes in the data
        self.__data = self.__data.iloc[1:] # drop the first line which is NaN now
        return self.__data.copy()

    """
    Transforms gradient values into absolute ones.
    Param:
        startValues : Dataframe containing initial data row on which the gradient values
                      are cumulatively added upon row by row
    Returns:
        Deep copy of the absolutized data
    """
    def toAbsolute(self, startValues):
        self.__data = pd.concat([startValues, self.__data]) # prepend the start values
        cols = [c for c in self.__data.columns if c not in self.__timeCols]
        self.__data.loc[:, cols] = self.__data.loc[:, cols].cumsum() # adding up the changes cumulatively
        return self.__data.copy()

    """
    Remove outliers via Z-Score, interpolate the emerging data gaps
    Params:
        threshold : minimum number of standard deviations by which a value of a measurement has to be
                    away from the measurement's mean value to be considered being an outlier
        method    : interpolation technique to use, see for details:
                    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
    Returns:
        Deep copy of the cleaned data
    """
    def removeOutliers(self, threshold=3, method='linear'):
        cols = [c for c in self.__data.columns if c not in self.__timeCols]
        sub = self.__data.loc[:, cols] # pick the chosen subset of columns
        lim = np.abs((sub - sub.mean()) / sub.std(ddof=0)) < threshold # apply z-score
        self.__data.loc[:, cols] = sub.where(lim, np.nan) # replace outliers with NaN
        self.__data = self.__data.interpolate(method) # fill the NaN-gaps
        return self.__data.copy()

    """
    Calculates and returns the distance correlation for a given feature to all other features,
    for their mean sampled with given time frequency, and above given threshold value
    """
    def dCorr(self, feature, freq='hour', threshold=0.4):
        df = self.__data.copy()
        df['_'+freq] = self.__data[freq]
        df = df.groupby('_'+freq).mean()
        df = df.drop(columns=list(filter(lambda i: i != freq, self.__timeCols)))
        df = df.corr(method=dcor.distance_correlation, numeric_only=True)[feature]
        return df[df.values >= threshold]

    """
    Add columns as lagged features for given list of columns with given list of periods
    """
    def addLag(self, columns, periods):
        for c in columns:
            for i in periods:
                self.__data[c+'_lagged_'+str(i)] = self.__data[c].shift(periods=i)
        return self.__data.copy()

    """
    Add columns as rolling means for given list of columns with given list of windows at given shift
    """
    def addRollingMean(self, columns, windows, shift=0):
        for c in columns:
            for i in windows:
                self.__data[c+'_rollingMean_'+str(i)] = self.__data[c].shift(shift).rolling(i).mean()
        return self.__data.copy()


if __name__ == '__main__':
    print("Error: Don't call ConsumptionData.py directly, import it.")
