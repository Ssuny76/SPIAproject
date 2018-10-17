import numpy as np

def transactionCost(pv, pv_, a, a_, p, p_, n, rate): #transaction cost of one stock
    # pv : portfolio value of time 1
    # pv_ : portfolio value of time 2
    # a : action(weight) of time 1
    # a_ : action(weight) of time 2
    # p : price of stock in time 1
    # p_ : price of stock in time 2
    # n : time
    # rate : transaction cost rate
    #cost = abs(pv * a / p - pv_ * a_ / p_) * p_ * rate
    
    if pv * a * p_ > pv_ * a_ * p:
        _cost = (p_ * pv * a * rate / p - pv_ * a_ * rate) / (0.5 - a_ * rate)

    elif pv * a * p_ < pv_ * a_ * p:
        _cost = (pv_ * a_ * rate - p_ * pv * a * rate / p) / (0.5 + a_ * rate)
    else:
        _cost = 0
    return _cost

def stateDiscretize(s):
    if float(s) > 0.30:
        s = str(0.30)
    if float(s) < -0.30:
        s = str(-0.30)
    return s

def benchmark(t, budget, price0, rate):
    # t : the interval of time transaction occurs
    # budget
    # price0 : price data
    # rate : transaction cost rate
    n = 1
    history = np.zeros(len(price0))
    history[0] = budget
    history[1] = budget
    pv = budget #portfolio value
    p = price0.iloc[1,2]  #p : price of stock 1 in the beginning
    _p = price0.iloc[1,4]   #_p : price of stock 2 in the beginning
    if t == 0: #no transaction
        while n < len(price0) - 1:
            n += 1
            s1_p = price0.iloc[n,2]
            s2_p = price0.iloc[n,4]
            _pv = (s1_p / p + s2_p / _p) * budget / 2
            history[n] = _pv
        return history


    else:    
        while n < len(price0) -1:
            n += 1
            s1_p = price0.iloc[n,2]
            s2_p = price0.iloc[n,4]
            pv_ = (s1_p / p + s2_p / _p) * pv / 2 

            if n-1 >= t and (n-1) % t == 0:
                s1_pri = price0.iloc[n-t,2] #prior bought price of stock1
                s2_pri = price0.iloc[n-t,4] #prior bought price of stock2
                tc = transactionCost(history[n-t], pv_, 0.5, 0.5, s1_pri, s1_p ,n, rate)
                pv = pv_ - tc
                history[n] = pv
                p = s1_p
                _p = s2_p
            else:
                history[n] = pv_

        return history

'''
    n = 1
    history = np.zeros(len(price0))
    history[0] = budget
    history[1] = budget
    pv = budget #portfolio value
    p = price0.iloc[0,2]  #p : price of stock 1 in the beginning
    _p = price0.iloc[0,4]   #_p : price of stock 2 in the beginning
    if t == 0: #no transaction
        while n < len(price0):
            s1_p = price0.iloc[n,2]
            s2_p = price0.iloc[n,4]
            pv = budget * 0.5 / p * s1_p + budget * 0.5 / _p * s2_p
            history[n] = pv
            n += 1
        return history

    else:    
        while n < len(price0):
            s1_p = price0.iloc[n,2]
            s2_p = price0.iloc[n,4]

            pv = budget * 0.5 / p * s1_p + budget * 0.5 / _p * s2_p
            
            if n % t == 0:
                s1_pri = price0.iloc[n-t,2] #prior bought price
                s2_pri = price0.iloc[n-t,4] #prior bought price
                transaction_cost = abs(pv / s1_p - budget / s1_pri) * 0.5 * s1_p * rate + \
                                    abs(pv / s2_p - budget / s2_pri) * 0.5 * s2_p * rate
                pv = pv - transaction_cost
                p = s1_p
                _p = s2_p
                budget = pv
            history[n] = pv
            n += 1
        return history
        '''