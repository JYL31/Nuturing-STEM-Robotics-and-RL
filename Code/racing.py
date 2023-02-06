# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:08:32 2023

@author: Jiayuan Liu
"""

import numpy as np
import random
import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam


class DQN_Agent:

    def __init__(self, observation_space = 1, action_space = 1, exploration_rate = 1, 
                 exploration_decay = 0.9999, learning_rate = 0.001, 
                 discount_factor = 0.95, memory_size = 10000, 
                 batch_size = 128):

        self.action_space = action_space
        self.observation_space = observation_space
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        self.RAM = deque(maxlen=int(memory_size))
        self.ROM = deque(maxlen=int(memory_size*0.1))
        
        # Deep Q learning network, input size is number of observation, output size is number of actions
        self.model = Sequential()
        self.model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96,96,3)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(216, activation='relu'))
        self.model.add(Dense(len(action_space), activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

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
            return random.randrange(len(self.action_space))
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    # train the deep Q learning network using past memory
    def experience_replay(self):
        memory = self.RAM + self.ROM
        if len(memory) < self.batch_size:
            return
        batch = random.sample(memory, self.batch_size)
        
        state = np.zeros((128,96,96,3))#(self.batch_size, self.observation_space)
        next_state = np.zeros((128,96,96,3))
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
    
def train_network():
    env = gym.make("CarRacing-v2", continuous=True)#, render_mode='human')
    observation_space = env.observation_space.shape
    action_space = [(-1, 1, 0), (0, 1, 0), (1, 1, 0), #           Action Space Structure
                    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2)] #        (Steering Wheel, Gas, Break)
                    #(-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2)] # Range        -1~1       0~1   0~1
    agent = DQN_Agent(observation_space, action_space)
    run = 0
    while run < 100:
        run += 1
        state = env.reset()
        total_rewards = 0
        while True:
            #env.render()
            action = agent.policy(state)
            state_next, reward, terminal, info = env.step(action_space[action])
            total_rewards += reward
            agent.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                verbose = "Episodes: " + str(run) + ", Exploration: " + str(agent.exploration_rate) + ", Reward: " + str(total_rewards) + '\n'
                print(verbose)
                break
            agent.experience_replay()

train_network()
