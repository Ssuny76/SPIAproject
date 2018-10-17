from agent import *
from functions import *
import sys
import time
from keras.utils import plot_model
import matplotlib.pyplot as plt   # Import matplotlib
import pandas as pd

start = time.time()


if len(sys.argv) != 5:
    print("Usage: python train.py [stock1] [stock2] [window] [episodes]")
    exit()

stock1_name, stock2_name, window_size, episode_count = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

agent = Agent(2 * window_size - 2)
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
history_set = []
half_benchmark = []
action_history = []
v =np.zeros(l+1)
v[0] = budget 

for e in range(episode_count):
    #initial weight 0 :1
    w = 0 
    _w = 1
    state = getState(data, _data, 0, window_size)
    total_profit = 0
    agent.history = []
    w_vec = []
    
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

            #remember
        agent.append_sample(state, action, reward, next_state, done)       
        state = next_state
    
        if e == 0:
            v[t+1] = v[t] * ((data[t+1] / data[t]) * 0.5 + (_data[t+1] / _data[t]) * 0.5)
            v[t+1] = v[t+1] - transactionCost(v[t], v[t+1], 0.5, 0.5, data[t], data[t+1], 0.002)
        """print("time " + str(t) + " : " + stock1_name + str(w) + " " + stock2_name + str(_w)) """
            
        if done:
            print("--------------------------------")
            print("Episode: " + str(e+1))
            print("--------------------------------")
            print("Total Portfolio Value: " + formatPrice(agent.history[-1]))
        
    if len(agent.memory) > agent.batch_size:
        agent.train_model()
        
    history_set.append(agent.history[-1])
    half_benchmark.append(v[-1])
    action_history.append(w_vec)
            
    agent.model.save("progress\models/model_ep_{}_{}_{}_{}_".format(str(stock1_name),str(stock2_name),window_size,episode_count) + str(e))
        
        
action_history = pd.DataFrame(action_history, index=np.arange(1,len(action_history)+1))
action_history.to_csv("progress/results/action_{}_{}_{}_{}.csv".format(str(stock1_name),str(stock2_name),window_size,episode_count))

#summary of created model
print(agent.model.summary())
"""plot_model(agent.model, to_file='progress\model_plot_%s_%d.png', show_shapes=True, show_layer_names=True) % (window_size, episode_count)"""

ep = np.argmax(history_set)

#print result as file
a = pd.DataFrame({'DQN':history_set,'half_benchmark':half_benchmark},index=np.arange(1,episode_count+1),columns=['DQN','half_benchmark'])       

a.to_csv("progress/results/{}_{}_{}_{}.csv".format(str(stock1_name),str(stock2_name),window_size,episode_count))

print(time.time()-start)

#plot the total_profit over episode
plt.subplot(2,1,1)
plt.plot(list(np.arange(1,episode_count+1)),history_set)
plt.xlabel('episode')
plt.ylabel('portfolio value')

plt.subplot(2,1,2)
plt.plot(np.arange(1,l+2,1), agent.history, label = 'DQN')
plt.plot(np.arange(1,l+2,1), v, label = 'benchmark')
plt.xlabel('time')
plt.ylabel('portfolio value')
plt.legend()
plt.tight_layout()
plt.savefig('progress/results/results_{}_{}_{}_{}.png'.format(str(stock1_name),str(stock2_name),window_size, episode_count))
plt.show()
