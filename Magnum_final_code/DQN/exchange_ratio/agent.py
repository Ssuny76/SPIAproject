import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size # normalized previous days
        self.action_size = 11 # [0,0.1,...,0.9,1]
        self.memory = deque(maxlen=10000) #similar to list except it doesn't have the fixed length
        self.model_name = model_name
        self.is_eval = is_eval #evaluation y/n
        self.gamma = 0.95
        self.epsilon = 1.0 
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.tau = 0.125
        self.learning_rate = 0.0005

        self.model = load_model("q-trader1/models/" + model_name) if is_eval else self._model()
        self.target_model = self._model()

    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu")) 
        model.add(Dense(units=128, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(units=64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(units=28, activation="relu"))
        model.add(Dropout(0.75))
        model.add(Dense(units=self.action_size))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate)) 

        return model

    def act(self, state):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        if not self.is_eval and np.random.rand() <= self.epsilon: #Random values in a given shape. random samples from a uniform distribution over [0, 1).
            return random.randrange(self.action_size) #randomly choose action

        act_values = self.model.predict(state) #Generates output predictions for the input samples.
        return np.argmax(act_values[0]) #act_values[0] looks like this: [0.67, 0.2]

    '''def remember(self, state, action, reward, next_state, done):
       self.memory.append((state, action, reward, next_state, done))'''

    #minibatch randomly chosen show better performance?
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
            target_f = self.target_model.predict(state)
            target_f[0][action] = target
            self.model.fit(state,target_f,epochs=1,verbose=0)    

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)
