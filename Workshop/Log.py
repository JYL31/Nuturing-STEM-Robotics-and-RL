# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:43:09 2023

@author: Jiayuan Liu
"""

import os
import csv

class csv_log:
    
    def __init__(self):
        cwd = os.getcwd()
        path = os.path.join(cwd, 'training_log.csv')

        self.file = open(path, 'w', newline='')
        header = ['Episode Number', 'Cart Position', 'Pole Angle', 'Pole Angular Velocity', 'Action', 'Exploration Rate', 'Reward', 'Score']
        self.writer = csv.writer(self.file)
        self.writer.writerow(header)
    
    def write(self, episode, state, action, exploration_rate, reward, total_reward):
        self.writer.writerow([episode, state[0][0], state[0][1], state[0][2], action, exploration_rate, reward, total_reward])
        
    def close(self):
        self.file.close()
    
    