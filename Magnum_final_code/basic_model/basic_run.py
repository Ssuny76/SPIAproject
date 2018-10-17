import time
from send_brain import QLearningTable
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from function import*

start_time = time.time()
#change the input file and name of the stock
price0=pd.read_csv('data/DATA/TRAIN/observationKlacSkx1y.csv')
stock_name = 'KS1'

if __name__=="__main__":
    RL=QLearningTable(actions = ['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'], initial =True)
    #RL=QLearningTable(actions = ['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'])
episodes = 1
budget = 10000
lr = 0.0001

ep_results = []
half_bench = []
firstSet_bench = []
action = []

new = 0
interval = 1
half_value = []
firstSet_value = []
trans_rate = 0.002 #transaction cost rate for every activity

for episode in range(1,episodes+1):

    w = np.zeros(len(price0))
    a0 = np.zeros(len(price0)) #action list
    a0[0] = 0.5
    m = np.arange(1,len(price0)+1,1)
    w[0] = budget ; w[1] = budget
    n = 1
    print(str(episode)+'/' + str(episodes))
    while n < len(price0)-1:
        s1 = price0.iloc[n-1:n+1,2]
        s1_change = round((s1.iloc[1]-s1.iloc[0])/s1.iloc[0],2)
        stateDiscretize(s1_change)
        s2 = price0.iloc[n-1:n+1,4]
        s2_change = round((s2.iloc[1]-s2.iloc[0])/s2.iloc[0],2)
        stateDiscretize(s2_change)
        s = ','.join([str(s1_change),str(s2_change)])

        s_1 = price0.iloc[n:n+2,2]
        s_1_change = round((s_1.iloc[1]-s_1.iloc[0])/s_1.iloc[0],2)
        stateDiscretize(s_1_change)
        s_2 = price0.iloc[n:n+2,4]
        s_2_change = round((s_2.iloc[1]-s_2.iloc[0])/s_2.iloc[0],2)
        stateDiscretize(s_2_change)
        s_ = ','.join([str(s_1_change),str(s_2_change)])

        s_ != 'terminal'

        a1 = RL.choose_action(s)
        a = float(a1)

        reward = (s_1_change * a + s_2_change * (1-a))* w[n] - \
                 transactionCost(w[n-1], w[n], a0[n-1], a, s1.iloc[0], s1.iloc[1], n, trans_rate)
        w[n+1] = w[n] + reward
        a0[n] = a
        RL.learn(s,a1,reward,s_)
        s = s_
        n = n+1
    half_value = benchmark(interval, budget, price0, trans_rate)
    firstSet_value = benchmark(0, budget, price0, trans_rate)
    action.append(a0)
    
    ep_results.append(w[len(price0)-1])
    half_bench.append(half_value[len(price0)-1])
    firstSet_bench.append(firstSet_value[len(price0)-1])

    print("episode " + str(episode) + ":" + str(w[len(price0)-1]))    
    if n == len(price0)-1:
        s_ = str('terminal')
        a1 = RL.choose_action(s)
        a = float(a1)
        a0[n] = a
        s = s_
        n = n+1

print(half_value)
print(firstSet_value)
action = pd.DataFrame(action)
action.to_csv('results2/action_{}.csv'.format(stock_name))

print('processing time :' + format(time.time()-start_time))
a = pd.DataFrame({'q-learning':ep_results,'half_bench':half_bench, 'firstSet_bench':firstSet_bench},index=np.arange(1,episodes+1),columns=['q-learning','half_bench', 'firstSet_bench'])       
a.to_csv('results2/results_{}.csv'.format(stock_name))

b = pd.DataFrame({'half_value':half_value,'firstSet_value':firstSet_value},index = np.arange(0,len(price0)),columns=['half_value','firstSet_value'])
b.to_csv('results2/benchmark_history_{}.csv'.format(interval))

#plot
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(list(np.arange(1,episodes+1)),ep_results)
plt.xlabel('episodes')
plt.ylabel('portfolio value')
plt.subplot(2,1,2)
plt.plot(m,w,label='q-learning')
plt.plot(m,half_value,label='benchmark(interval_{})'.format(interval))
#plt.plot(m,firstSet_value, label = 'firstSet_benchmark')
plt.xlabel('days')
plt.ylabel('portfolio value')
plt.legend()
plt.show()
fig.savefig('plot2/q_learning_{}.jpg'.format(stock_name))
RL.q_table.to_csv('q_tables2/q_table{}.csv'.format(stock_name))