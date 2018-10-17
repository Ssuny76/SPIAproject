from agent import *
from functions import *
import sys
import time
from forex_python.converter import CurrencyRates
from keras.utils import plot_model
import matplotlib.pyplot as plt   # Import matplotlib
import pandas as pd

start = time.time()

if len(sys.argv) != 8:
    print(sys.argv)
    print(len(sys.argv))
    print("Usage: python train.py [stock1] [stock2] [window] [episodes] [stock1_currency] [stock2_currency] [budget_currency]")
    exit()

stock1_name, stock2_name, window_size, episode_count, stock1_currency, stock2_currency, budget_currency = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), str(sys.argv[5]), str(sys.argv[6]),str(sys.argv[7]) 

agent = Agent(2 * window_size)
date = getDateVec(stock1_name) #date
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
history_set = []
half_benchmark = []
action_history = []
v =np.zeros(l)
v[0] = budget 

for e in range(episode_count):
    print("Episode " + str(e+1) + "/" + str(episode_count))
    #initial weight 0.5 :0.5
    w = 0.5 
    _w = 0.5
    state = getState(data, _data, 0, window_size, w, _w)
    total_profit = 0
    agent.history = []
    w_vec = []
    
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
                    if budget_currency == stock1_currency:
                        conversion_rate_start = get_conversion_rate(stock1_currency,stock2_currency,t,date)
                        if t==0:
                            conversion_rate_end = conversion_rate_start
                        else:
                            conversion_rate_end = get_conversion_rate(stock1_currency,stock2_currency,t-1,date)
                        print("conversion rate start:", conversion_rate_start)
                        print("conversion rate end:", conversion_rate_end)
                        stock1_currency_portfolio_value = agent.history[t-1] * ((data[t] / data[t-1]) * w  + (_data[t] / _data[t-1]) * _w * conversion_rate_start/conversion_rate_end)
                        print ("portfolio_value in stock1_currency:", stock1_currency_portfolio_value)
                        portfolio_value = stock1_currency_portfolio_value
                        agent.history.append(portfolio_value)
                        reward = agent.history[t] - agent.history[t-1]
                        print ("reward:", reward)
                        total_profit += reward
                    else:
                        conversion_rate_start = get_conversion_rate(stock2_currency,stock1_currency,t,date)
                        if t==0:
                            conversion_rate_end = conversion_rate_start
                        else:
                            conversion_rate_end = get_conversion_rate(stock2_currency,stock1_currency,t-1,date)
                        print("conversion rate start:", conversion_rate_start)
                        print("conversion rate end:", conversion_rate_end)
                        stock2_currency_portfolio_value = agent.history[t-1] * ((data[t] / data[t-1]) * w * conversion_rate_start/conversion_rate_end + (_data[t] / _data[t-1]) * _w)
                        print ("portfolio_value in stock1_currency:", stock2_currency_portfolio_value)
                        portfolio_value = stock2_currency_portfolio_value
                        agent.history.append(portfolio_value)
                        reward = agent.history[t] - agent.history[t-1]
                        print ("reward:", reward)
                        total_profit += reward
        next_state = getState(data, _data, t + 1, window_size, w, _w)
        w_vec.append(w)

        #terminal state
        done = True if t == l - 1 else False

        #remember
        agent.memory.append((state, action, reward, next_state, done))       
        state = next_state
    
        if e == 0:
            if t >= 1:
                v[t]=v[t-1] * ((data[t] / data[t-1]) * 0.5 + (_data[t] / _data[t-1]) * 0.5)

        """print("time " + str(t) + " : " + stock1_name + str(w) + " " + stock2_name + str(_w)) """
        if done:
            print("--------------------------------")
            print("Episode: " + str(e+1))
            print("--------------------------------")
            print("Total Portfolio Value: " + formatPrice(agent.history[l-1]))

    history_set.append(agent.history[-1])
    half_benchmark.append(v[l-1])
    action_history.append(w_vec)
    
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    agent.target_train()

            
    if e % 10 == 0:
        agent.model.save("q-trader1/models/model_ep_{}_{}_{}_{}_".format(str(stock1_name),str(stock2_name),window_size,episode_count) + str(e))
        
        
action_history = pd.DataFrame(action_history, index=np.arange(1,len(action_history)+1))
action_history.to_csv("q-trader1/action_{}_{}_{}_{}.csv".format(str(stock1_name),str(stock2_name),window_size,episode_count))

#summary of created model
print(agent.model.summary())
"""plot_model(agent.model, to_file='q-trader1\model_plot_%s_%d.png', show_shapes=True, show_layer_names=True) % (window_size, episode_count)"""



#print result as file
a = pd.DataFrame({'DQN':history_set,'half_benchmark':half_benchmark},index=np.arange(1,episode_count+1),columns=['DQN','half_benchmark'])       

a.to_csv("q-trader1/{}_{}_{}_{}.csv".format(str(stock1_name),str(stock2_name),window_size,episode_count))

print(time.time()-start)

#plot the total_profit over episode
plt.subplot(2,1,1)
plt.plot(list(np.arange(1,episode_count+1)),history_set)
plt.xlabel('episode')
plt.ylabel('portfolio value')

plt.subplot(2,1,2)
plt.plot(np.arange(1,l+1,1), agent.history, label = 'DQN')
plt.plot(np.arange(1,l+1,1), v, label = 'benchmark')
plt.xlabel('time')
plt.ylabel('portfolio value')
plt.legend()
plt.tight_layout()
plt.savefig('q-trader1/results_{}_{}_{}_{}.png'.format(str(stock1_name),str(stock2_name),window_size, episode_count))
plt.show()
