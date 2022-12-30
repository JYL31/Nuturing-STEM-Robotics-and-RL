# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:14:39 2022

@author: Jiayuan Liu
"""

import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os


# Reference: https://github.com/gsurma/cartpole/blob/master/cartpole.py
class DQN_Agent:

    def __init__(self, observation_space = 1, action_space = 1, exploration_rate = 1, 
                 exploration_decay = 0.999, learning_rate = 0.001, 
                 discount_factor = 0.95, memory_size = 2000, 
                 batch_size = 64, layer_units = [64, 32, 16],
                 load = False):

        self.action_space = action_space
        self.observation_space = observation_space
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.layer_units = layer_units
        
        self.RAM = deque(maxlen=int(memory_size))
        self.ROM = deque(maxlen=int(memory_size*0.1))
        
        if load == True:
            self.model = load_model('DQN_model')
        else:
        # Deep Q learning network, input size is number of observation, output size is number of actions
            self.model = Sequential()
            self.model.add(Dense(self.layer_units[0], input_shape=(self.observation_space,), activation="relu", kernel_initializer='he_uniform'))
            self.model.add(Dense(self.layer_units[1], activation="relu", kernel_initializer='he_uniform'))
            self.model.add(Dense(self.layer_units[2], activation="relu", kernel_initializer='he_uniform'))
            self.model.add(Dense(self.action_space, activation="linear", kernel_initializer='he_uniform'))
            self.model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        
        exists = os.path.exists('model_plot.png')
        if exists:
            os.remove('model_plot.png')
        tf.keras.utils.plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # memory of the agent
    def remember(self, state, action, reward, next_state, done):
        self.RAM.append((state, action, reward, next_state, done))
        if len(self.ROM) != self.memory_size*0.1:
          self.ROM.append((state, action, reward, next_state, done))

    # how agent decides to take an action:
    # randomly picks a number between 0 and 1, if less than exploration rate, agent chooses a random action
    # else, agent chooses the one with the largest Q value
    def policy(self, state):
      if np.random.rand() < self.exploration_rate:
        return random.randrange(self.action_space)
      q_values = self.model.predict(state, verbose=0)
      return np.argmax(q_values[0])

    # train the deep Q learning network using past memory
    def experience_replay(self):
        memory = self.RAM + self.ROM
        if len(memory) < self.batch_size:
            return
        batch = random.sample(memory, self.batch_size)
        
        state = np.zeros((self.batch_size, self.observation_space))
        next_state = np.zeros((self.batch_size, self.observation_space))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            next_state[i] = batch[i][3]
            done.append(batch[i][4])
        # do batch prediction to save speed
        target = self.model.predict(state, verbose=0)
        target_next = self.model.predict(next_state, verbose=0)

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_next[i]))

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)
        
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(0.01, self.exploration_rate)
    
    def save(self):
        self.model.save('DQN_model')

    def get_model(self):
        return self.model