import numpy as np
import pandas as pd




class QLearningTable:
    #actions=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    #price0=pd.read_csv('observation30new.csv')
    #def __init__(self,actions,learning_rate=0.1,reward_decay=0,'''epsilon_decay=0.99,epsilon_min=0.01,e_greedy=0.9,'''e_greedy=0.01):
    def __init__(self,actions,learning_rate=0.1,reward_decay=0,e_greedy=0.01):
        self.actions=actions
        self.lr=learning_rate
        self.gamma=reward_decay
        self.epsilon=e_greedy
        #self.epsilon_decay=epsilon_decay
        #self.epsilon_min=epsilon_min
        #self.q_table=pd.DataFrame(columns=self.actions,dtype=np.float64)
        #self.q_table = pd.read_csv('q_tablefixingtransaction10.csv', index_col = 0)
        self.q_table = pd.read_csv('q_tablePARAMETER88.csv', index_col = 0) 

        
    def generalize_state(self,price_change_rate):
        if price_change_rate>0.3:
            price_change_rate=round(price_change_rate,1)
        if price_change_rate<-0.3:
            price_change_rate=round(price_change_rate,1)
        return price_change_rate

    def transactionCost(self,pv, pv_, a, a_, p, p_, n, rate): #transaction cost of one stock
    # pv : portfolio value of time 1
    # pv_ : portfolio value of time 2
    # a : action(weight) of time 1
    # a_ : action(weight) of time 2
    # p : price of stock in time 1
    # p_ : price of stock in time 2
    # n : time
    # rate : transaction cost rate
        cost = abs(pv * a / p - pv_ * a_ / p_) * p_ * rate

        return cost

            

    def choose_action(self,observation):
        self.check_state_exist(observation)
        if np.random.uniform()>self.epsilon:
            state_action=self.q_table.loc[observation,:]
            state_action=state_action.reindex(np.random.permutation(state_action.index))
            action=state_action.idxmax()
            #print(action)
            '''if self.epsilon >= self.epsilon_min:
                self.epsilon *= self.epsilon_decay'''
        else:
            action=np.random.choice(self.actions)
            '''if self.epsilon >= self.epsilon_min:
                self.epsilon *= self.epsilon_decay'''
        return action

    def learn(self,s,a,r,s_):
        self.check_state_exist(s_)
        self.q_table.fillna(0,inplace=True)
        q_predict=self.q_table.loc[s,a]
        if s_!='terminal':
            q_target=r+self.gamma*self.q_table.loc[s_,:].max()
        else:
            q_target=r
        self.q_table.loc[s,a]+=self.lr*(q_target-q_predict)

    def benchmark(self, t, budget, price0, rate):
    # t : the interval of time transaction occurs
    # budget
    # price0 : price data
    # rate : transaction cost rate
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

    def check_state_exist(self,state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state
                    )
                )
