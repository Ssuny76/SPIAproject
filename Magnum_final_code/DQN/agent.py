import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

class Agent:
    def __init__(self, state_size, is_eval=False,  model_name=""):
        self.state_size = state_size # normalized previous days
        self.action_size = 11 
        self.memory = deque(maxlen=5000) #similar to list except it doesn't have the fixed length
        self.model_name = model_name
        self.is_eval = is_eval #evaluation y/n
        self.batch_size = 64
        self.gamma = 0.95
        self.learning_rate = 0.0005
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = load_model("progress/models/" + model_name) if is_eval else self._model()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu")) 
        model.add(Dense(units=128, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=28, activation="relu"))
        model.add(Dropout(0.75))
        model.add(Dense(units=self.action_size, activation = 'linear'))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0, amsgrad=False)) 

        return model

    def act(self, state):
        if not self.is_eval and np.random.rand() <= self.epsilon: #Random values in a given shape. random samples from a uniform distribution over [0, 1).
            return random.randrange(self.action_size) #randomly choose action

        act_values = self.model.predict(state) #Generates output predictions for the input samples.
        return np.argmax(act_values[0]) #act_values[0] looks like this: [0.67, 0.2]

    '''def remember(self, state, action, reward, next_state, done):
       self.memory.append((state, action, reward, next_state, done))'''


    def train_model(self):
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        mini_batch = random.sample(self.memory, self.batch_size)

        """errors = np.zeros(self.batch_size)"""
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][1][0]
            actions.append(mini_batch[i][1][1])
            rewards.append(mini_batch[i][1][2])
            next_states[i] = mini_batch[i][1][3]
            dones.append(mini_batch[i][1][4])

        # 현재 상태에 대한 모델의 큐함수
        target = self.model.predict(states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            old_val = target[i][actions[i]]
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * (np.amax(self.model.predict(next_states)[i]))

        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)
        
        return mini_batch

            
    def append_sample(self, state, action, reward, next_state, done):
        error = 0
        self.memory.append([error, [state, action, reward, next_state, done]])
    

