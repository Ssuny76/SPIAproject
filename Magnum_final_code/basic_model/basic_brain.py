import numpy as np
import pandas as pd
######READ ME######
#initialize = True if q_table is not made in advance, otherwise, False
#change the input Q-table

class QLearningTable:
    def __init__(self,actions,learning_rate=0.0001,reward_decay=0.9,epsilon=0.9,epsilon_decay=0.99,epsilon_min=0.01, initial=False):
        self.actions=actions
        self.lr=learning_rate
        self.gamma=reward_decay
        self.epsilon=epsilon
        self.epsilon_decay=epsilon_decay
        self.epsilon_min=epsilon_min
        #self.q_table = self.initialize
        self.initial = initial
        self.q_table = pd.read_csv('q_tables2/q_tableMP9.csv', index_col = 0) if not self.initial else pd.DataFrame(columns=self.actions,dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0]*len(self.actions),index=self.q_table.columns,name=state))

    def learn(self,s,a,r,s_):
        self.check_state_exist(s_)
        q_predict=self.q_table.loc[s,a]
        if s_!='terminal':
            q_target=r+self.gamma*self.q_table.loc[s_,:].max()
        else:
            q_target=r
        self.q_table.loc[s,a]+=self.lr*(q_target-q_predict) 
        
        
    def choose_action(self,observation):
        self.check_state_exist(observation)
        if np.random.uniform()>self.epsilon:
            state_action=self.q_table.loc[observation,:]
            state_action=state_action.reindex(np.random.permutation(state_action.index))
            action=state_action.idxmax()
            if self.epsilon >= self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        else:
            action=np.random.choice(self.actions)
            if self.epsilon >= self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        return action
