"""
forecast.py
"""

import ConsumptionData as cp
import modelLSTM as ml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


if __name__ == '__main__':

    url = "http://kammeyer.uk:8086"
    token = "CPSJ6xw1U72IcJjfLgzaukP24o1CL3grIQuvaw-Zq1MK9htUYNPwFUdKEalwl2-xMHFrVKOgG8tRFLgIkoneBw=="
    org = "591d9e9c3fc5e3ee"
    startTime = '2025-06-18T00:00:00Z'
    endTime   = '2025-06-27T01:00:00Z'
    data = cp.ConsumptionData()
    data.setConnection(url, token, org)

    print("=== Fetched data for " + startTime + " to " + endTime + " ===")
    data.fetch(startTime, endTime)

    actualData = cp.ConsumptionData()
    actualData.setConnection(url, token, org)
    actualData.fetch('2025-06-27T00:00:00Z', '2025-06-28T01:00:00Z')
    actual = actualData.get()
    model = ml.modelLSTM("lstm_absolute_3y_21_06_2025")
    forecasted = model.forecast(data, 'price')

    print(actual)

    plt.figure(figsize=(12, 8))
    plt.plot(pd.to_datetime(actual['_time']), actual['price'], label='Actual')
    plt.plot(pd.to_datetime(actual['_time']), forecasted, label='Forecasts')
    plt.legend()
    plt.title("Actual vs Forecasts")
    plt.savefig('actual-vs-forecasts.png')

    mae_rf = mean_absolute_error(actual['price'], forecasted)
    max = actual['price'].max()
    min = actual['price'].min()
    print("max: " + str(max) + " / " + "min: " + str(min))
    print("Mean Absolute Error (MAE): ", mae_rf)
    print("Normalized MAE: ", mae_rf / (max - min))
