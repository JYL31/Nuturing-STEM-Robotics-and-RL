# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:43:09 2023

@author: Jiayuan Liu
"""

import os
import csv

class csv_log:
    
    def __init__(self, file_type):
        cwd = os.getcwd()
        path = os.path.join(cwd,'Logs')
        if os.path.exists(path) == False:
            os.mkdir(path)
        if file_type == 'train':
            path = os.path.join(path,'training_log.csv')
        elif file_type == 'test':
            path = os.path.join(path,'testing_log.csv')
        if os.path.exists(path) == True:
            os.remove(path)

        self.file = open(path, 'w', newline='')
        header = ['Episode Number','Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity', 'Action', 'Reward', 'Total Reward']
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)
    
    def write(self, episode, state, action, reward, total_reward):
        self.writer.writerow([episode, state[0][0], state[0][1], state[0][2], state[0][3], action, reward, total_reward])
        
    def close(self):
        self.file.close()
    
    