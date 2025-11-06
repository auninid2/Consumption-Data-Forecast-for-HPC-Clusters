"""
training.py
"""

import ConsumptionData as cp
import modelLSTM as ml

import numpy as np
import pandas as pd
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning) # suppress misleading warnings
from pathlib import Path


if __name__ == '__main__':

    modelName = "lstm_absolute_3y_21_06_2025"
    data = cp.ConsumptionData()

    datafile = Path("./3y-data-2022.06.21-2025.06.21.csv")
    if datafile.is_file(): # check on existence
        data.fromCSV(datafile)
    else:
        url = "http://kammeyer.uk:8086"
        token = "CPSJ6xw1U72IcJjfLgzaukP24o1CL3grIQuvaw-Zq1MK9htUYNPwFUdKEalwl2-xMHFrVKOgG8tRFLgIkoneBw=="
        org = "591d9e9c3fc5e3ee"
        startTime = '2022-06-21T00:00:00Z'
        endTime   = '2025-06-21T00:00:00Z'
        data.setConnection(url, token, org)
        data.fetch(startTime, endTime)
        data.toCSV(datafile)

    print("=== Fetched data for " + data.get().loc[:,'_time'].iloc[0].strftime('%Y-%m-%d %H:%M:%S') + " to " + data.get().loc[:,'_time'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S') + " ===")
    print(data.get())

    print("=== Adding lagged price values ===")
    data.addLag(['price'], list(range(1,168)))
    print("=== Adding rolling mean values ===")
    data.addRollingMean(['price'], list(range(2,168)), shift=1)

    data.dropNAN()

    print("=== Execute Distance Correlation Analysis for 'price' ===")
    data.dropCols(['fossilFuelPercentage', 'fossilEnergy'])
    relevantCols = data.dCorr('price', threshold=0.8).index

    print("=== Keeping only relevant data columns according to Distance Correlation ===")
    data.keepCols(relevantCols)
    print(data.get())

    featureCols = [x for x in data.get().columns.to_list() if x != 'price' and x != '_time']

    with open(modelName + "_training_features.txt", 'w') as outfile:
        outfile.write('\n'.join(s for s in featureCols))

    model = ml.modelLSTM(modelName)
    model.train(data, featureCols, 'price')
