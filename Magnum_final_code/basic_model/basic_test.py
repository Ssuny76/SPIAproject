
# coding: utf-8

# In[ ]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from function import *

q_table = pd.read_csv('q_tables2/q_tableNS10.csv',index_col = 0)
data = pd.read_csv('data/DATA/TEST/observationAmatCaj1y.csv')
model_name = 'NS10_AC_action'
w = np.zeros(len(data))
bench = np.zeros(len(data))
a0 = np.zeros(len(data)) #action list
a0[0] = 0.5
budget = 10000
w[0] = 10000 ; w[1] = 10000
bench[0] = 10000 ; bench[1] = 10000
m = np.arange(1,len(data)+1,1)
n = 1
random = 0
interval = 1
trans_rate = 0.002
while n < len(data)-1:
    s1 = data.iloc[n-1:n+1,2]
    s1_change = str(round((s1.iloc[1]-s1.iloc[0])/s1.iloc[0],2))
    stateDiscretize(s1_change)
    s2 = data.iloc[n-1:n+1,4]
    s2_change = str(round((s2.iloc[1]-s2.iloc[0])/s2.iloc[0],2))
    stateDiscretize(s2_change)
    s = ','.join([s1_change,s2_change])
    if s in q_table.index:
        action=q_table.loc[s,:]
        action=float(np.argmax(action))
        
    else:
        action = w[n-1]*a0[n-1]*data.iloc[n,2]/(w[n]*data.iloc[n-1,2])
        #action = np.random.choice([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
        random = random + 1
    a0[n] = float(action)
    s_1 = data.iloc[n:n+2,2]
    s_1_change = round((s_1.iloc[1]-s_1.iloc[0])/s_1.iloc[0],2)
    stateDiscretize(s_1_change)
    s_2 = data.iloc[n:n+2,4]
    s_2_change = round((s_2.iloc[1]-s_2.iloc[0])/s_2.iloc[0],2)
    stateDiscretize(s_2_change)
    w[n+1] = (float(s_1_change) * action + float(s_2_change) * (1-action))* w[n] + w[n]
    tran_cost1 = transactionCost(w[n], w[n+1], a0[n-1], a0[n], data.iloc[n,2], data.iloc[n+1,2], n, trans_rate)
    w[n+1] = w[n+1] - tran_cost1
    n += 1 
bench = benchmark(interval, budget, data, trans_rate)
bench2 = benchmark(0, budget, data, trans_rate)
    
dic = pd.DataFrame({'q-learning': w, 'benchmark(0.5:0.5)': bench})
dic.to_csv('q_learning_test.csv')

action = pd.DataFrame(a0)
action.to_csv('results2/test_action_{}.csv'.format(model_name))
pv = pd.DataFrame(w)
pv.to_csv('results2/test_pv_{}.csv'.format(model_name))

print(random)
fig = plt.figure()
plt.plot(m, w, label = 'q-learning')
plt.plot(m, bench, label = 'benchmark(0.5,0.5)_interval1')
#plt.plot(m, bench2, label = 'benchmark(0.5,0.5)_firstSet')
plt.legend()
plt.show()
plt.xlabel('days')
plt.ylabel('price')
fig.savefig('plot2/test_{}.jpg'.format(model_name))
    
        
        

