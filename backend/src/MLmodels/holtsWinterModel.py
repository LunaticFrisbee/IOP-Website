import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import base64

def getGraph(forecast, testData):
    plt.figure().set_figwidth(15)
    plt.plot(range(len(testData)), testData)
    plt.plot(range(len(forecast)), forecast)
    plt.ylabel('Load Demand')
    plt.xlabel('Hours')
    plt.legend(['load', 'forecast'])
    plt.savefig('forecast.png', bbox_inches='tight')
    with open('forecast.png', mode='rb') as file:
        img = file.read()
    return (json.dumps(base64.encodebytes(img).decode('utf-8')))

# predictions = holt_winters_forecast(series, alpha, beta, gamma, h, n_preds)
def holt_winters_forecast(series, alpha, beta, gamma, h, n_preds):
    level, trend = series[0], series[1] - series[0]
    seasonals = initial_seasonality(series, h)
    result = [series[0]]
    for i in range(1, len(series) + n_preds):
        if i < h:
            result.append(series[i])
        elif i>=h and i <len(series) :
            val = series[i]
            last_level=level
            level = alpha * (val - seasonals[i % h]) + (1 - alpha) * (last_level + trend)
            trend = beta * (level - last_level) + (1 - beta) * trend
            seasonals[i % h] = gamma * (val - level) + (1 - gamma) * seasonals[i % h]
            result.append(level + trend + seasonals[i % h])
        else:
            m = i - len(series) + 1
            result.append((level + trend*m) + seasonals[i % h])
    return result

def initial_seasonality(series, h):
    seasonals = np.zeros(h)
    seasonal_avg=0
    for i in range(h):
        seasonal_avg += series[i]
    seasonal_avg/=h
    print(seasonal_avg)
    for i in range(h):
        print(series[i])
        seasonals[i] = (series[i] - seasonal_avg)/h
    seas = pd.DataFrame(seasonals, columns=['seasonals'])
    seas.to_csv("electrical_load_seasonals.csv", index=False)
    return seasonals


def holtWinterModel(dataSet, npreds):
    forecast = holt_winters_forecast(dataSet['load'][:-npreds], 0.07, 0.5, 0.7, 12, 24)
    plot = getGraph(forecast, dataSet['load'][:-npreds])
    return {"forecast": json.dumps(forecast[-npreds:]), "plot": plot}