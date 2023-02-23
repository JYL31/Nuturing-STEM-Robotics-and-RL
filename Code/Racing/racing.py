# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 14:08:32 2023

@author: Jiayuan Liu
"""

import numpy as np
import random
import gym
import cv2
import csv
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from custom_racing import CarRacing

class DQN_Agent:

    def __init__(self, observation_space = 1, action_space = 1, exploration_rate = 1, 
                 exploration_decay = 0.9995, learning_rate = 0.001, 
                 discount_factor = 0.95, memory_size = 5000, 
                 batch_size = 64):

        self.action_space = action_space
        self.observation_space = observation_space
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        self.memory = deque(maxlen=int(memory_size))
        
        # Deep Q learning network, input size is number of observation, output size is number of actions
        self.model = Sequential()
        self.model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=3, activation='relu', input_shape=(96,96,5)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=12, kernel_size=(4, 4), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(216, activation='relu'))
        self.model.add(Dense(len(action_space), activation=None))
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))

    # memory of the agent
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

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
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        
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
    
def train_network():
    path = 'E:\Melb Uni\Capstone\Racing'

    env = gym.wrappers.RecordVideo(CarRacing(continuous=True, render_mode='rgb_array'), 
                                       video_folder=path, name_prefix='train', episode_trigger = lambda x: x % 2 == 0)
    
    observation_space = env.observation_space.shape

    action_space = [(0, 1, 0), (-1, 0, 0), (0, 0, 0), #           Action Space Structure
                    (1, 0, 0), (0, 0, 0.5)]           #        (Steering Wheel, Gas, Break)
                                                      # Range        -1~1       0~1   0~1
                                                      
    agent = DQN_Agent(observation_space, action_space)
    run = 0
    file = open('E:/Melb Uni/Capstone/Racing/results.csv', 'w', newline='')
    writer = csv.writer(file)

    while run < 100:
        run += 1
        state = env.reset(seed=10)
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = state.astype(float)
        state /= 255.0
        frame_stack = deque([state]*5, maxlen=5)
        total_rewards = 0
        while True:
            #env.render()
            cur_frame_stack = np.array(frame_stack)
            cur_frame_stack = np.transpose(cur_frame_stack, (1, 2, 0))
            action = agent.policy(cur_frame_stack)
            
            reward = 0
            for i in range(3):
                state_next, r, terminal, info = env.step(action_space[action])
                reward += r
                if terminal:
                    break
        
            if action_space[action][1] == 1 and reward > 0:
                reward = 1.2*reward
            total_rewards += reward
            
            state_next = cv2.cvtColor(state_next, cv2.COLOR_BGR2GRAY)
            state_next = state_next.astype(float)
            state_next /= 255.0
            frame_stack.append(state_next)
            next_frame_stack = np.array(frame_stack)
            next_frame_stack = np.transpose(next_frame_stack, (1, 2, 0))
            
            agent.remember(cur_frame_stack, action, reward, next_frame_stack, terminal)
            
            if terminal or total_rewards < 0:
                verbose = "Episodes: " + str(run) + ", Exploration: " + str(agent.exploration_rate) + ", Score: " + str(total_rewards) + '\n'
                print(verbose)
                writer.writerow([agent.exploration_rate, total_rewards])
                
                break
            agent.experience_replay()
    file.close()

train_network()

