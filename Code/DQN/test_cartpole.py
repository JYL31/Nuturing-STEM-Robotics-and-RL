# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:14:39 2022
@author: Jiayuan Liu

"""

import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from cartpole import CartPoleEnv


class DQN_Agent:
    """
    Title: Cartpole_DQN source code
    Author: pythonlessons
    Date: 2020
    Availability: https://github.com/pythonlessons/Reinforcement_Learning/blob/master/01_CartPole-reinforcement-learning/Cartpole_DQN.py
    """
    
    def __init__(self, observation_space = 1, action_space = 1, exploration_rate = 1, 
                 learning_rate = 0.001):

        self.action_space = action_space
        self.observation_space = observation_space
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = 0.95
        self.batch_size = 64
        self.exploration_decay = 0.999
        self.memory_size = 2000
        
        self.RAM = deque(maxlen=int(self.memory_size))
        self.ROM = deque(maxlen=int(self.memory_size*0.1))
        
        # MLP Network model
        self.model = Sequential()
        self.model.add(Dense(64, input_shape=(self.observation_space,), activation="relu", kernel_initializer='he_uniform'))
        self.model.add(Dense(32, activation="relu", kernel_initializer='he_uniform'))
        self.model.add(Dense(16, activation="relu", kernel_initializer='he_uniform'))
        self.model.add(Dense(self.action_space, activation="linear", kernel_initializer='he_uniform'))
        self.model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))

    # memory of the agent
    def remember(self, state, action, reward, next_state, done):
        self.RAM.append((state, action, reward, next_state, done))
        if len(self.ROM) != self.memory_size*0.1: # set 10% of memory to permanently store early experinces
          self.ROM.append((state, action, reward, next_state, done))

    # how agent decides to take an action:
    # randomly picks a number between 0 and 1, if less than exploration rate, agent chooses a random action
    # else, agent chooses the one with the largest Q value
    def policy(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    # update the deep Q learning network using past memory
    def experience_replay(self):
        memory = self.RAM + self.ROM
        if len(memory) < self.batch_size: # only update when there's enough past experiences for a batch
            return
        batch = random.sample(memory, self.batch_size)
        state = np.zeros((self.batch_size, self.observation_space))
        next_state = np.zeros((self.batch_size, self.observation_space))
        
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            next_state[i] = batch[i][3]
            done.append(batch[i][4])
        
        # batch prediction for Q(s,a) and Q(s',a)
        target = self.model.predict(state, verbose=0)
        target_next = self.model.predict(next_state, verbose=0)

        for i in range(self.batch_size):
            # target Q-value
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Bellman Equation
                # Q*(s,a) = r + gamma * max(Q(s',a))
                target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_next[i]))

        # train and update the network
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)
        
        # reduce exploration rate
        self.exploration_rate *= self.exploration_decay
        self.exploration_rate = max(0.01, self.exploration_rate)
    
def train():
    
    # initialize an envrionment
    env = CartPoleEnv()
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    
    # initilaize an agent
    agent = DQN_Agent(observation_space = observation_space, action_space = action_space)
        
    run = 0
    done = False
    while run < 100 and done == False:
        # reset environment to initial state
        run += 1
        state = env.reset(seed=5)
        state = np.reshape(state, [1, observation_space])
        total_rewards = 0
        while True:
            action = agent.policy(state) # get action
            state_next, reward, terminal, _, _ = env.step(action) # perform the action
            reward = -100*(abs(state_next[1]) - abs(state[0][1])) # modify the reward
            state_next = np.reshape(state_next, [1, observation_space])
            agent.remember(state, action, reward, state_next, terminal) # store experience
            state = state_next
            total_rewards += 1 
            if terminal:
                verbose = "Episodes: " + str(run) + ", Exploration: " + str(agent.exploration_rate) + ", Score: " + str(total_rewards) + '\n'
                print(verbose)
                break
            agent.experience_replay() # update network

train()
