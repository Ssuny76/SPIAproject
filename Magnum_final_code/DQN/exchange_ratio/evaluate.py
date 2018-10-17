import keras
from keras.models import load_model

from agent import Agent
from functions import *
import sys
import pandas as pd
import matplotlib.pyplot as plt   # Import matplotlib

if len(sys.argv) != 4:
    print("Usage: python evaluate.py [stock1] [stock2] [model]")
    exit()

stock1_name, stock2_name, model_name = sys.argv[1], sys.argv[2], sys.argv[3]
model = load_model("q-trader1/models/" + model_name)
window_size = (model.layers[0].input.shape.as_list()[1])/2

agent = Agent(2 * window_size, True, model_name) #state_size, evaluation, model name
data = getStockDataVec(stock1_name) #stock1 data
_data = getStockDataVec(stock2_name) #stock2 data
l = len(data) - 1
_l = len(_data) -1
if l != _l : #two stock data should be equal on the start date
    if l > _l:
        data = data[:_l]
    else:
        _data = _data[:l]

batch_size = 32
portfolio_value = 0
budget = 10000

w = 0.5
_w = 0.5

state = getState(data, _data, 0, window_size, w, _w) #(data, t, n)

total_profit = 0
agent.history = []
half_benchmark = []

#benchmark portfolio values
v =np.zeros(l)
v[0] = budget
v[1] = budget

for t in range(l):
    action = agent.act(state)
    reward = 0
    if t == 0:
        w = action/10
        _w = (10 - action)/10
        portfolio_value = budget
        agent.history.append(portfolio_value)		
    else:
        for i in range(0,11):
            if action == i:
                w = i/10
                _w = (10 - i)/10
                portfolio_value = agent.history[t-1] * (data[t] / data[t-1] * w + _data[t] / _data[t-1] * _w)
                agent.history.append(portfolio_value)

                reward = agent.history[t] - agent.history[t-1]
                total_profit += reward
    if t > 1:
        v[t] = v[t-1] + v[t-1] * ((data[t] - data[t-1]) / data[t-1] * 0.5 + (_data[t] - _data[t-1]) / _data[t-1] * 0.5)

    print(stock1_name + ":" + str(w) + ", " + stock2_name + ":" + str(_w))
    print("time : "+ str(t) + '| gain : '+ str(reward) + '| total profit : '+ str(total_profit))

    next_state = getState(data, _data, t + 1, window_size, w, _w)

    #terminal state
    if t == l-1:
        done = True
    else:
        done = False

    #remember
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state

    if done:
        print("--------------------------------")
        print("Total Profit: " + formatPrice(total_profit))
        print("--------------------------------")
        half_benchmark.append(v[l-1])


#print result as file
a = pd.DataFrame({'DQN':agent.history[l-1],'half_bench':v[l-1]},index=np.arange(1,2),columns=['DQN','half_bench'])       
a.to_csv('q-trader1/results_{}_eval.csv'.format(window_size)) 

#plot the total_profit over episode
plt.plot(np.arange(1,l+1,1), agent.history, label = 'DQN')
plt.plot(np.arange(1,l+1,1), v, label = 'benchmark')
plt.legend()
plt.show()
plt.savefig('DQN_%s_eval.png' %(window_size)) 
