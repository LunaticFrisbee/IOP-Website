import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from matplotlib import pyplot as plt
from fastapi.responses import FileResponse
import json
import base64

def getGraph(forecast, testData):
    plt.figure().set_figwidth(15)
    plt.plot(range(len(testData)), testData)
    plt.plot(range(len(testData)), forecast)
    plt.ylabel('Load Demand')
    plt.xlabel('Hours')
    plt.legend(['load', 'forecast'])
    plt.savefig('forecast.png', bbox_inches='tight')
    with open('forecast.png', mode='rb') as file:
        img = file.read()
    return (json.dumps(base64.encodebytes(img).decode('utf-8')))


def holtWinterModel(dataSet, npreds):
    model = ExponentialSmoothing(dataSet['load'][:-npreds], trend='add', seasonal='add', seasonal_periods=24)
    model_fit = model.fit()
    forecast = model_fit.forecast(npreds)
    plot = getGraph(forecast, dataSet['load'][-npreds:]); 
    return {"forecast": forecast[-npreds:].to_json(), "plot": plot}