import numpy as np
import math

budget = 10000

# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataVec(stock):
    vec = []

    # read stock1 data
    lines = open("progress/" + stock + ".csv", "r").read().splitlines()
    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))

    return vec

# returns the sigmoid - later step : open/close price ratio
'''def sigmoid(x):
    return 1 / (1 + math.exp(-x))'''

# returns an an n-day state representation ending at time t
def getState(data, _data, t, n):
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
    return np.array([res])


def transactionCost(pv, pv_, a, a_, p, p_, rate): #transaction cost of one stock
    # pv : portfolio value of time 1
    # pv_ : portfolio value of time 2
    # a : action(weight) of time 1
    # a_ : action(weight) of time 2
    # p : price of stock in time 1
    # p_ : price of stock in time 2
    # rate : transaction cost rate
    if pv * a * p_ > pv_ * a_ * p:
        _cost = ((p_ * pv * a * rate / p) -( pv_ * a_ * rate)) / (0.5 - a_ * rate)
    elif pv * a * p_ < pv_ * a_ * p:
        _cost = ((pv_ * a_ * rate) - (p_ * pv * a * rate / p)) / (0.5 + a_ * rate)
    else:
        _cost = 0
    return _cost
