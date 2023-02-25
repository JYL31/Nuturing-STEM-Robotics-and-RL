# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:43:09 2023

@author: Jiayuan Liu
"""

import os
import csv

class csv_log:
    
    def __init__(self, env, file_type):
        cwd = os.getcwd()
        self.env = env
        if self.env == 0:
            path = os.path.join(cwd, r'Cartpole\Logs')
        elif self.env == 1:
            path = os.path.join(cwd, r'CarRacing\Logs')
        if os.path.exists(path) == False:
            os.makedirs(path)
        if file_type == 'train':
            path = os.path.join(path,'training_log.csv')
        elif file_type == 'test':
            path = os.path.join(path,'testing_log.csv')
        if os.path.exists(path) == True:
            os.remove(path)

        self.file = open(path, 'w', newline='')
        if self.env == 0:
            header = ['Episode Number', 'Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity', 'Action', 'Reward', 'Score']
        elif self.env == 1:
            header = ['Episode Number', 'Action', 'Reward', 'Score']
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)
    
    def write(self, episode, state, action, reward, total_reward):
        if self.env == 0:
            self.writer.writerow([episode, state[0][0], state[0][1], state[0][2], state[0][3], action, reward, total_reward])
        elif self.env == 1:
            self.writer.writerow([episode, action, reward, total_reward])
        
    def close(self):
        self.file.close()
    
    