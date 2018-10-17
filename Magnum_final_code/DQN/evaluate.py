import keras
from keras.models import load_model

from agent import Agent
from functions import *
import sys
import pandas as pd
import matplotlib.pyplot as plt   # Import matplotlib

if len(sys.argv) != 5:
    print("Usage: python evaluate.py [stock1] [stock2] [model] [episode]")
    exit()

stock1_name, stock2_name, model_name, episodes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
model = load_model("progress/models/" + model_name)
window_size = (model.layers[0].input.shape.as_list()[1])/2 + 1

agent = Agent(2 * window_size - 2, True, model_name) #state_size, evaluation, model name
data = getStockDataVec(stock1_name) #stock1 data
_data = getStockDataVec(stock2_name) #stock2 data
l = len(data) - 1
_l = len(_data) -1
if l != _l : #two stock data should be equal on the start date
    if l > _l:
        data = data[:_l]
    else:
        _data = _data[:l]


portfolio_value = 0
budget = 10000

w = 0
_w = 1

state = getState(data, _data, 0, window_size) #(data, t, n)


agent.history = []
w_vec = []
w_vec.append(0)

#benchmark portfolio values
v =np.zeros(l+1)
v[0] = budget

for t in range(l):
    if t == 0:
        action = w
        portfolio_value = budget
        agent.history.append(portfolio_value)
        portfolio_value = agent.history[t] * ((data[t+1] / data[t]) * w + (_data[t+1] / _data[t]) * _w)
        agent.history.append(portfolio_value)
        reward = agent.history[1] - agent.history[0]
    else:
        w_past = w
        action = agent.act(state)
        w = action / 10
        _w = 1 - w            
        before_value = agent.history[t] * ((data[t+1] / data[t]) * w + (_data[t+1] / _data[t]) * _w)
        portfolio_value = before_value - transactionCost(portfolio_value, before_value, w_past, w, data[t], data[t+1], 0.002)
        agent.history.append(portfolio_value)
        reward = agent.history[t+1] - agent.history[t]          

    next_state = getState(data, _data, t+1, window_size)
    w_vec.append(w)
    
    done = True if t == l - 1 else False  
    state = next_state
    
    v[t+1] = v[t] * ((data[t+1] / data[t]) * 0.5 + (_data[t+1] / _data[t]) * 0.5)
    v[t+1] = v[t+1] - transactionCost(v[t], v[t+1], 0.5, 0.5, data[t], data[t+1], 0.002)

    print(stock1_name + ":" + str(w) + ", " + stock2_name + ":" + str(_w))
    print("time : "+ str(t+1) + '| gain : '+ str(reward) + '| total profit : '+ str(agent.history[-1]))

    if done:
        print("--------------------------------")
        print("Portfolio Value: " + formatPrice(agent.history[-1]))
        print("--------------------------------")
        print("Benchmark: " + formatPrice(v[-1]))


action_history = pd.DataFrame(w_vec, index=np.arange(l+1))
action_history.to_csv("progress/results/action_{}_{}_{}_eval.csv".format(str(stock1_name),str(stock2_name),window_size))              
              
#print result as file
a = pd.DataFrame({'DQN':agent.history,'half_bench':v},index = np.arange(l+1), columns = ['DQN','half_bench'])       
a.to_csv('progress/results/results_{}_{}_{}_eval.csv'.format(str(stock1_name),str(stock2_name),window_size)) 

#plot the total_profit over episode
plt.plot(np.arange(1,l+2,1), agent.history, label = 'DQN')
plt.plot(np.arange(1,l+2,1), v, label = 'benchmark')
plt.xlabel('time')
plt.ylabel('portfolio value')
plt.title(episodes + " episodes")
plt.legend()
plt.savefig('progress/results/results_{}_{}_{}_eval.png'.format(str(stock1_name),str(stock2_name),episodes))
plt.show()

