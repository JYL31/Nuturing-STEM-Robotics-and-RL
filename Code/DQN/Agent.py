# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:14:39 2022
@author: Jiayuan Liu

Title: Cartpole_DQN source code
Author: pythonlessons
Date: 2020
Availability: https://github.com/pythonlessons/Reinforcement_Learning/blob/master/01_CartPole-reinforcement-learning/Cartpole_DQN.py

"""

import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os


class DQN_Agent:

    def __init__(self, observation_space = 1, action_space = 1, exploration_rate = 1, 
                 learning_rate = 0.001, memory_type = 1, env = 0, load = False):

        self.env = env
        self.action_space = action_space
        self.observation_space = observation_space
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = 0.95
        self.memory_type = memory_type
        self.batch_size = 64
        
        if self.env == 0:
            self.exploration_decay = 0.999
            self.memory_size = 2000
        elif self.env == 1:
            self.exploration_decay = 0.9995
            self.memory_size = 5000
        
        if self.memory_type == 1:
            self.RAM = deque(maxlen=int(self.memory_size))
            self.ROM = deque(maxlen=int(self.memory_size*0.1))
        else:
            self.RAM = deque(maxlen=int(self.memory_size))
            self.ROM = deque(maxlen=0)
        
        cwd = os.getcwd()
        if env == 0:
            path = r'Cartpole\DQN_model'
            path = os.path.join(cwd, path)
            if os.path.exists(path) == False:
                os.mkdir(path)
            if load == True:
                self.model = load_model(path)
            else:
                self.model = Sequential()
                self.model.add(Dense(64, input_shape=(self.observation_space,), activation="relu", kernel_initializer='he_uniform'))
                self.model.add(Dense(32, activation="relu", kernel_initializer='he_uniform'))
                self.model.add(Dense(16, activation="relu", kernel_initializer='he_uniform'))
                self.model.add(Dense(self.action_space, activation="linear", kernel_initializer='he_uniform'))
                self.model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))
        elif env == 1:
            path = r'CarRacing\DQN_model'
            path = os.path.join(cwd, path)
            if os.path.exists(path) == False:
                os.mkdir(path)
            if load == True:
                self.model = load_model(path)
            else:
                self.model = Sequential()
                self.model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96,96,5)))
                self.model.add(MaxPooling2D(pool_size=(2, 2)))
                self.model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
                self.model.add(MaxPooling2D(pool_size=(2, 2)))
                self.model.add(Flatten())
                self.model.add(Dense(216, activation='relu'))
                self.model.add(Dense(len(self.action_space), activation=None))
                self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        
        path = os.path.join(path, 'model_plot.png')
        exists = os.path.exists(path)
        if exists:
            os.remove(path)
        tf.keras.utils.plot_model(self.model, to_file=path, show_shapes=True, show_layer_names=True)

    # memory of the agent
    def remember(self, state, action, reward, next_state, done):
        self.RAM.append((state, action, reward, next_state, done))
        if len(self.ROM) != self.memory_size*0.1 and self.memory_type != 0:
          self.ROM.append((state, action, reward, next_state, done))

    # how agent decides to take an action:
    # randomly picks a number between 0 and 1, if less than exploration rate, agent chooses a random action
    # else, agent chooses the one with the largest Q value
    def policy(self, state):
        if np.random.rand() < self.exploration_rate:
            if self.env == 0:
                return random.randrange(self.action_space)
            elif self.env == 1:
                return random.randrange(len(self.action_space))
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    # train the deep Q learning network using past memory
    def experience_replay(self):
        memory = self.RAM + self.ROM
        if len(memory) < self.batch_size:
            return
        batch = random.sample(memory, self.batch_size)
        if self.env == 0:
            state = np.zeros((self.batch_size, self.observation_space))
            next_state = np.zeros((self.batch_size, self.observation_space))
        elif self.env == 1:
            state = np.zeros((64,96,96,5))
            next_state = np.zeros((64,96,96,5))
        action, reward, done = [], [], []

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
        cwd = os.getcwd()
        if self.env == 0:
            path = os.path.join(cwd, r'Cartpole\DQN_model')
            self.model.save(path)
        elif self.env == 1:
            path = os.path.join(cwd, r'CarRacing\DQN_model')
            self.model.save(path)

    def get_model(self):
        return self.model