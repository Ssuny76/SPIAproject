import numpy as np
import math
import csv
import datetime
from decimal import *
from forex_python.converter import CurrencyRates

budget = 10000

# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# get the date from csv file
def getDateVec(stock):
    dates=[]
    lines = open("q-trader1/" + stock + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        newline = str(line.split(",")[0].replace("-"," "))
        yy = int(newline[0:4])
        mm = int(newline[newline.find(' ')+1:newline.find(' ')+3])
        dd = int(newline[newline.find(' ')+4:])
        date = datetime.datetime(yy,mm,dd)
        dates.append(date)
    return dates

def get_conversion_rate(original,final,t,date):
    c = CurrencyRates()
    r=c.get_rate(original,final,date[t])
    return r

# converting currency rate  (conversion rate is different everyday)
def convert(original,final,price,t,date):
    c = CurrencyRates()
    r=c.convert(original,final,price,date[t]) #converse the reward 
    return r 

# returns the vector containing stock data from a fixed file
def getStockDataVec(stock):
    vec = []
    # read stock1 data
    lines = open("q-trader1/" + stock + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        line0 = line.split(",")[4]
        if line0 != '':
            vec.append(float(line0))

    return vec

# returns the sigmoid - later step : open/close price ratio
'''def sigmoid(x):
    return 1 / (1 + math.exp(-x))'''

# returns an an n-day state representation ending at time t
def getState(data, _data, t, n, w, _w):
    d = int(t+1)- int(n)
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1] 
    # pad with t0 #(if d>=0 price of 10 days, otherwise data of time 0 replicate and then from 0 to t)
    _block = _data[d:t + 1] if d >= 0 else -d * [_data[0]] + _data[0:t + 1]
    res = []
    #res vector : first n are stock1, later n are stock2
    for i in range(int(n-1)):
        res.append((block[i + 1] - block[i])/block[i])
    for i in range(int(n-1)):
        res.append((_block[i + 1] - _block[i])/_block[i])
    res.append(w)
    res.append(_w)
    return np.array([res])
