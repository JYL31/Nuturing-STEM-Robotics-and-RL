# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:53:18 2022

@author: Jiayuan Liu
"""

import gym
import os
import cv2
import pygame
import tkinter as tk
import customtkinter as ctk
import numpy as np
from collections import deque
from custom_racing import CarRacing
from statistics import mean
from util_play import play
from Agent import DQN_Agent
from Log import csv_log
from Log_Veiwer import log_viewer
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

class display:
    
    def __init__(self, root, fig):
        self.container = root
        self.container.title("Basic GUI Layout with Grid")
        self.container.maxsize(1500,  800)
        self.fig = fig

    def setup(self):
        self.create_widgets()
        self.setup_layout()  

    def create_widgets(self):
        self.leftFrame = ctk.CTkFrame(self.container, width=100, height=780)
        self.midFrame = ctk.CTkFrame(self.container, width=300, height=780)
        self.rightFrame = ctk.CTkFrame(self.container, width=400, height=800)
        self.rightFrame1 = ctk.CTkFrame(self.rightFrame, width=400, height=380)
        self.rightFrame2 = ctk.CTkFrame(self.rightFrame, width=400, height=380)
        
        self.label_exr = ctk.CTkLabel(self.leftFrame, text='Exploration rate : ')
        self.var_exr = tk.DoubleVar(value=1)
        self.rb_exr1 = ctk.CTkRadioButton(self.leftFrame, text='1', variable=self.var_exr, value=1)
        self.rb_exr2 = ctk.CTkRadioButton(self.leftFrame, text='0.1', variable=self.var_exr, value=0.1)
        
        self.label_lr = ctk.CTkLabel(self.leftFrame, text='Learning rate : ')
        self.var_lr = tk.DoubleVar(value=0.001)
        self.rb_lr1 = ctk.CTkRadioButton(self.leftFrame, text='0.001', variable=self.var_lr, value=0.001)
        self.rb_lr2 = ctk.CTkRadioButton(self.leftFrame, text='0.1', variable=self.var_lr, value=0.1)
        
        self.label_env = ctk.CTkLabel(self.leftFrame, text='Envrionment : ')
        self.var_env = tk.DoubleVar(value=0)
        self.rb_env1 = ctk.CTkRadioButton(self.leftFrame, text='Cartpole', variable=self.var_env, value=0)
        self.rb_env2 = ctk.CTkRadioButton(self.leftFrame, text='Car Racing', variable=self.var_env, value=1)
        
        self.var_mem = tk.IntVar(value=1)
        self.rb_mem = ctk.CTkCheckBox(self.leftFrame, text='permanently store initial experience', variable=self.var_mem, onvalue=1, offvalue=1)
        
        self.but_train = ctk.CTkButton(self.leftFrame, text='Start Training', command=self.train_network)
        self.but_test = ctk.CTkButton(self.leftFrame, text='Start Testing', command=self.test_network)
        self.but_reset = ctk.CTkButton(self.leftFrame, text='Reset to Default', command=self.reset)
        self.but_save = ctk.CTkButton(self.leftFrame, text='Save Network Model', command=self.save)
        self.but_load = ctk.CTkButton(self.leftFrame, text='Load Network Model', command=self.load)
        self.but_log = ctk.CTkButton(self.leftFrame, text='Log', command=self.open_csv)
        self.but_play = ctk.CTkButton(self.leftFrame, text='Play', command=self.manual_play)
        
        self.label_nn = ctk.CTkLabel(self.midFrame, text='Deep Q Network Structure')
        self.canvas_nn = ctk.CTkCanvas(self.midFrame, width=340, height=700)
    
        self.label_plot = ctk.CTkLabel(self.rightFrame1, text='Reward over episodes')
        self.canvas_plot = FigureCanvasTkAgg(self.fig, self.rightFrame1)
        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, self.rightFrame1)
        
        self.label_txt = ctk.CTkLabel(self.rightFrame2, text='Verbose')
        self.txtbox = ctk.CTkTextbox(self.rightFrame2, width=400, height=390)
        
        self.radiobuttons = [self.rb_exr1, self.rb_exr2, self.rb_lr1, self.rb_lr2,
                             self.rb_env1, self.rb_env2]
        
        self.buttons = [self.but_train, self.but_test, self.but_reset,
                        self.but_save, self.but_load, self.but_log,
                        self.but_play]
        
    def setup_layout(self):
        self.leftFrame.grid(row=0, column=0, padx=10, pady=5)
        self.midFrame.grid(row=0, column=1, padx=10, pady=5)
        self.rightFrame.grid(row=0, column=2, padx=10, pady=5)
        self.rightFrame.grid_propagate(False)
        
        self.label_exr.grid(row=0, column=0, padx=5, pady=5)
        self.rb_exr1.grid(row=0, column=1, padx=5, pady=5)
        self.rb_exr2.grid(row=0, column=2, padx=5, pady=5)
        
        self.label_lr.grid(row=1, column=0, padx=5, pady=5)
        self.rb_lr1.grid(row=1, column=1, padx=5, pady=5)
        self.rb_lr2.grid(row=1, column=2, padx=5, pady=5)
        
        self.label_env.grid(row=2, column=0, padx=5, pady=5)
        self.rb_env1.grid(row=2, column=1, padx=5, pady=5)
        self.rb_env2.grid(row=2, column=2, padx=5, pady=5)
        
        self.rb_mem.grid(row=3, column=1, padx=5, pady=5)
        
        self.but_train.grid(row=4, column=1, padx=5, pady=5)
        self.but_test.grid(row=5, column=1, padx=5, pady=5)
        self.but_reset.grid(row=6, column=1, padx=5, pady=5)
        self.but_save.grid(row=7, column=1, padx=5, pady=5)
        self.but_load.grid(row=8, column=1, padx=5, pady=5)
        self.but_log.grid(row=9, column=1, padx=5, pady=5)
        self.but_play.grid(row=10, column=1, padx=5, pady=5)
        
        self.label_nn.grid(row=0, column=0, padx=5, pady=5)
        self.canvas_nn.grid(row=1, column=0, padx=5, pady=5)
        
        self.rightFrame1.pack(side='top')
        self.rightFrame1.pack_propagate(False)
        self.rightFrame2.pack(side='bottom')
        self.rightFrame2.pack_propagate(False)
        
        self.label_plot.pack(side='top')
        self.canvas_plot.get_tk_widget().pack(side='bottom')
        self.toolbar.pack(side='bottom')
        self.canvas_plot.draw()
        
        self.label_txt.pack(side='top')
        self.txtbox.pack(side='bottom')
    
    def save(self):
        self.agent.save()
            
    def load(self):
        self.agent = DQN_Agent(load=True)
        self.image = tk.PhotoImage(file = 'model_plot.png')
        self.canvas_nn.create_image(172, 265, image=self.image)
        self.canvas_nn.update()
        
        self.but_train.configure(state="disabled")
        for i in self.radiobuttons:
            i.configure(state="disabled")

    def reset(self):
        self.var_exr.set(value=1)
        self.var_lr.set(value=0.001)
        self.var_env.set(value=0)
        self.var_mem.set(value=1)
        self.canvas_nn.delete("all")
        self.canvas_plot.get_tk_widget().delete("all")
        self.txtbox.delete("1.0", "end")
        
        for i in self.buttons:
            i.configure(state="normal")
        for i in self.radiobuttons:
            i.configure(state="normal")
        self.rb_mem.configure(state="normal")
        
        if isinstance(self.agent, DQN_Agent):
            del self.agent
    
    def animate(self):
        self.ax.clear()
        self.ax.plot(self.rewards)
        plt.pause(1)

    
    def train_network(self):
        
        for i in self.buttons:
            i.configure(state="disabled")
        for i in self.radiobuttons:
            i.configure(state="disabled")
        self.rb_mem.configure(state="disabled")
        
        self.rewards = []
        self.ax = self.fig.add_subplot(111)
        
        cwd = os.getcwd()
        
        if self.var_env.get() == 0:
            path = os.path.join(cwd, r'Cartpole\Videos')
            if os.path.exists(path) == False:
                os.makedirs(path)
            env = gym.wrappers.RecordVideo(gym.make('CartPole-v0', render_mode='rgb_array'), 
                                           video_folder=path, episode_trigger = lambda x: x % 2 == 0,
                                           name_prefix='train')
            env.seed(5)
            observation_space = env.observation_space.shape[0]
            action_space = env.action_space.n
            log = csv_log(env=0, file_type='train')
        elif self.var_env.get()==1:
            path = os.path.join(cwd, r'CarRacing\Videos')
            if os.path.exists(path) == False:
                os.makedirs(path)
            env = gym.wrappers.RecordVideo(CarRacing(continuous=True, render_mode='rgb_array'), 
                                           video_folder=path, episode_trigger = lambda x: x % 2 == 0,
                                           name_prefix='train')
            observation_space = [96, 96, 5]
            action_space = [(0, 1, 0), (-1, 0, 0), (0, 0, 0), #           Action Space Structure
                            (1, 0, 0), (0, 0, 0.5)]           #        (Steering Wheel, Gas, Break)
                                                              # Range        -1~1       0~1   0~1
            log = csv_log(env=1, file_type='train')
        self.agent = DQN_Agent(observation_space, action_space, exploration_rate=self.var_exr.get(), 
                               learning_rate=self.var_lr.get(), memory_type=self.var_mem.get(), 
                               env=self.var_env.get())
        if self.var_env.get() == 0:
            self.image = tk.PhotoImage(file = r'Cartpole\DQN_model\model_plot.png')
        elif self.var_env.get() == 1:
            self.image = tk.PhotoImage(file = r'CarRacing\DQN_model\model_plot.png')
        self.canvas_nn.create_image(172, 265, image=self.image)
        self.canvas_nn.update()
        
        run = 0
        done = False
        while run < 100 and done == False:
            run += 1
            if self.var_env.get() == 0:
                state = env.reset()
                state = np.reshape(state, [1, observation_space])
            elif self.var_env.get() == 1:
                state = env.reset(seed=10)
                state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
                state = state.astype(float)
                state /= 255.0
                frame_stack = deque([state]*5, maxlen=5)
            total_rewards = 0
            while True:
                #env.render()
                if self.var_env.get() == 0:
                    action = self.agent.policy(state)
                    state_next, reward, terminal, info = env.step(action)
                    reward = -100*(abs(state_next[2]) - abs(state[0][2]))
                    state_next = np.reshape(state_next, [1, observation_space])
                    self.agent.remember(state, action, reward, state_next, terminal)
                    state = state_next
                    total_rewards += 1
                
                elif self.var_env.get() == 1:
                    cur_frame_stack = np.array(frame_stack)
                    cur_frame_stack = np.transpose(cur_frame_stack, (1, 2, 0))
                    cur_frame_stack = np.expand_dims(cur_frame_stack, axis=0)
                    action = self.agent.policy(cur_frame_stack)
            
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
                    self.agent.remember(cur_frame_stack, action, reward, next_frame_stack, terminal)
                
                if terminal or total_rewards < 0:
                    verbose = "Episodes: " + str(run) + ", Exploration: " + str(self.agent.exploration_rate) + ", Score: " + str(total_rewards) + '\n'
                    print(verbose)
                    self.txtbox.insert(tk.END, verbose)
                    self.txtbox.update()
                    self.rewards.append(total_rewards)
                    self.animate()
                    self.canvas_plot.draw()
                    if self.var_env.get() == 0:
                        if run >= 5 and mean(self.rewards[-4:]) >= 195:
                            done = True
                    elif self.var_env.get() == 1:
                        if run >= 5 and mean(self.rewards[-4:]) >= 1000:
                            done = True
                    break
                log.write(run, state, action, reward, total_rewards)
                self.agent.experience_replay()
        self.agent.save()
        print("Training Completed!")
        env.close()
        log.close()
        for i in self.buttons:
            i.configure(state="normal")
        for i in self.radiobuttons:
            i.configure(state="normal")
        self.rb_mem.configure(state="normal")

    def test_network(self):
        for i in self.buttons:
            i.configure(state="disabled")
        for i in self.radiobuttons:
            i.configure(state="disabled")
        self.rb_mem.configure(state="disabled")
        
        cwd = os.getcwd()
        
        self.agent = DQN_Agent(load=True, env=self.var_env.get())
        model = self.agent.get_model()
                
        if self.var_env.get() == 0:
            path = os.path.join(cwd, r'Cartpole\Videos')
            if os.path.exists(path) == False:
                os.makedirs(path)
            env = gym.wrappers.RecordVideo(gym.make('CartPole-v0', render_mode='rgb_array'), 
                                           video_folder=path, episode_trigger = lambda x: True,
                                           name_prefix='train')
            env.seed(5)
            observation_space = env.observation_space.shape[0]
            action_space = env.action_space.n
            log = csv_log(env=0, file_type='test')
            self.image = tk.PhotoImage(file = r'Cartpole/DQN_model/model_plot.png')
        elif self.var_env.get()==1:
            path = os.path.join(cwd, r'CarRacing\Videos')
            if os.path.exists(path) == False:
                os.makedirs(path)
            env = gym.wrappers.RecordVideo(CarRacing(continuous=True, render_mode='rgb_array'), 
                                           video_folder=path, episode_trigger = lambda x: True,
                                           name_prefix='test')
            observation_space = [96, 96, 5]
            action_space = [(0, 1, 0), (-1, 0, 0), (0, 0, 0), #           Action Space Structure
                            (1, 0, 0), (0, 0, 0.5)]           #        (Steering Wheel, Gas, Break)
                                                              # Range        -1~1       0~1   0~1
            log = csv_log(env=1, file_type='test')
            self.image = tk.PhotoImage(file = r'CarRacing\DQN_model\model_plot.png')
        
        self.canvas_nn.create_image(172, 265, image=self.image)
        self.canvas_nn.update()
        
        for episodes in range(1,6):
            done = False
            if self.var_env.get() == 0:
                state = env.reset()
                state = np.reshape(state, [1, observation_space])
            elif self.var_env.get() == 1:
                state = env.reset(seed=10)
                state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
                state = state.astype(float)
                state /= 255.0
                frame_stack = deque([state]*5, maxlen=5)
            total_rewards = 0
            while not done:
                #env.render()
                if self.var_env.get() == 0:
                    action = np.argmax(model.predict(state, verbose=0))
                    state_next, reward, done, info = env.step(action)
                    reward = -100*(abs(state_next[2]) - abs(state[0][2]))
                    state = np.reshape(state_next, [1, observation_space])
                    total_rewards += 1
                elif self.var_env.get() == 1:
                    cur_frame_stack = np.array(frame_stack)
                    cur_frame_stack = np.transpose(cur_frame_stack, (1, 2, 0))
                    cur_frame_stack = np.expand_dims(cur_frame_stack, axis=0)
                    action = np.argmax(model.predict(cur_frame_stack, verbose=0))
            
                    reward = 0
                    for i in range(3):
                        state_next, r, done, info = env.step(action_space[action])
                        reward += r
                        if done:
                            break
        
                    if action_space[action][1] == 1 and reward > 0:
                        reward = 1.2*reward
                    total_rewards += reward
            
                    state_next = cv2.cvtColor(state_next, cv2.COLOR_BGR2GRAY)
                    state_next = state_next.astype(float)
                    state_next /= 255.0
                    frame_stack.append(state_next)
                    
                if done:
                    verbose = "Episodes: " + str(episodes) + ", Score: " + str(total_rewards) + '\n'
                    print(verbose)
                    self.txtbox.insert(tk.END, verbose)
                    self.txtbox.update()
                    break
                log.write(episodes, state, action, reward, total_rewards)
        env.close()
        log.close()
        for i in self.buttons:
            i.configure(state="normal")
        for i in self.radiobuttons:
            i.configure(state="normal") 
        self.rb_mem.configure(state="normal")
        
    def open_csv(self):
        
        def destroy():
            self.table_window.destroy()
        
        self.table_window = ctk.CTkToplevel(self.container)
        self.table_window.protocol("WM_DELETE_WINDOW", destroy)
        self.table_window.geometry("800x650")
        self.table_window.pack_propagate(False)
        self.table_window.resizable(0, 0)
        self.table_GUI = log_viewer(self.table_window)
        
    def manual_play(self):
        for i in self.buttons:
            i.configure(state="disabled")
        for i in self.radiobuttons:
            i.configure(state="disabled")
        self.rb_mem.configure(state="disabled")
        
        if self.var_env.get() == 0:
            mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
            play(gym.make("CartPole-v0"), fps = 15, keys_to_action = mapping, txtbox = self.txtbox)
        elif self.var_env.get() == 1:
            mapping = {(pygame.K_LEFT,): [-1, 0, 0], (pygame.K_RIGHT,): [1, 0, 0],
                       (pygame.K_UP,): [0, 1, 0], (pygame.K_DOWN,): [0, 0, 0.5]}
            play(CarRacing(continuous=True), fps = 15, keys_to_action = mapping,
                 txtbox = self.txtbox, seed=10, continuous=True)
        
        for i in self.buttons:
            i.configure(state = "normal")
        for i in self.radiobuttons:
            i.configure(state = "normal") 
        self.rb_mem.configure(state = "normal")
        



