# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:53:18 2022

@author: Jiayuan Liu
"""

import gym
import os
import pygame
import tkinter as tk
import customtkinter as ctk
import numpy as np
from statistics import mean
from util_play import play
from Agent import DQN_Agent
from Log import csv_log
from Log_Veiwer import log_viewer
from matplotlib import animation
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
        
        self.label_exr = ctk.CTkLabel(self.leftFrame, text='exploration rate : ')
        self.var_exr = tk.DoubleVar(value=1)
        self.rb_exr1 = ctk.CTkRadioButton(self.leftFrame, text='1', variable=self.var_exr, value=1)
        self.rb_exr2 = ctk.CTkRadioButton(self.leftFrame, text='0.1', variable=self.var_exr, value=0.1)
        
        self.label_lr = ctk.CTkLabel(self.leftFrame, text='learning rate : ')
        self.var_lr = tk.DoubleVar(value=0.001)
        self.rb_lr1 = ctk.CTkRadioButton(self.leftFrame, text='0.001', variable=self.var_lr, value=0.001)
        self.rb_lr2 = ctk.CTkRadioButton(self.leftFrame, text='0.1', variable=self.var_lr, value=0.1)
        
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
        
        self.radiobuttons = [self.rb_exr1, self.rb_exr2, self.rb_lr1, self.rb_lr2]
        
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
        
        self.rb_mem.grid(row=2, column=1, padx=5, pady=5)
        
        self.but_train.grid(row=3, column=1, padx=5, pady=5)
        self.but_test.grid(row=4, column=1, padx=5, pady=5)
        self.but_reset.grid(row=5, column=1, padx=5, pady=5)
        self.but_save.grid(row=6, column=1, padx=5, pady=5)
        self.but_load.grid(row=7, column=1, padx=5, pady=5)
        self.but_log.grid(row=8, column=1, padx=5, pady=5)
        self.but_play.grid(row=9, column=1, padx=5, pady=5)
        
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
        
        log = csv_log(file_type='train')
        
        self.rewards = []
        self.ax = self.fig.add_subplot(111)
        
        cwd = os.getcwd()
        path = os.path.join(cwd,'Videos')
        if os.path.exists(path) == False:
            os.mkdir(path)
            
        env = gym.wrappers.RecordVideo(gym.make('CartPole-v0', render_mode='rgb_array'), 
                                       video_folder=path, episode_trigger = lambda x: x % 2 == 0,
                                       name_prefix='train')
        env.seed(5)
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        self.agent = DQN_Agent(observation_space, action_space, exploration_rate=self.var_exr.get(), 
                               learning_rate=self.var_lr.get(), memory_type=self.var_mem.get())
        
        self.image = tk.PhotoImage(file = 'model_plot.png')
        self.canvas_nn.create_image(172, 265, image=self.image)
        self.canvas_nn.update()
        
        run = 0
        done = False
        while run < 100 and done == False:
            run += 1
            state = env.reset()
            state = np.reshape(state, [1, observation_space])
            step = 0
            while True:
                env.render()  
                action = self.agent.policy(state)
                state_next, reward, terminal, info = env.step(action)
                #reward = reward if not terminal else -reward
                reward = -100*(abs(state_next[2]) - abs(state[0][2]))
                state_next = np.reshape(state_next, [1, observation_space])
                self.agent.remember(state, action, reward, state_next, terminal)
                state = state_next
                step += 1
                if terminal:
                    verbose = "Episodes: " + str(run) + ", Exploration: " + str(self.agent.exploration_rate) + ", Score: " + str(step) + '\n'
                    print(verbose)
                    self.txtbox.insert(tk.END, verbose)
                    self.txtbox.update()
                    self.rewards.append(step)
                    self.animate()
                    self.canvas_plot.draw()
                    if run >= 5 and mean(self.rewards[-4:]) >= 195:
                        self.agent.save()
                        done = True
                        print("Training Completed!")
                    break
                log.write(run, state, action, reward, step)
                self.agent.experience_replay()
        env.close()
        log.close()
        #ani._stop()
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
        
        log = csv_log(file_type='test')
        
        cwd = os.getcwd()
        path = os.path.join(cwd,'Videos')
        if os.path.exists(path) == False:
            os.mkdir(path)
        
        self.agent = DQN_Agent(load=True)
        model = self.agent.get_model()
        
        self.image = tk.PhotoImage(file = 'model_plot.png')
        self.canvas_nn.create_image(172, 265, image=self.image)
        self.canvas_nn.update()
        
        env = gym.wrappers.RecordVideo(gym.make('CartPole-v0', render_mode='rgb_array'), 
                                       video_folder=path, episode_trigger = lambda x: True,
                                       name_prefix='test')
        observation_space = env.observation_space.shape[0]
        for episodes in range(1,6):
            state = env.reset()
            state = np.reshape(state, [1, observation_space])
            done = False
            step = 0
            while not done:
                env.render()
                action = np.argmax(model.predict(state, verbose=0))
                next_state, reward, done, _ = env.step(action)
                state = np.reshape(next_state, [1, observation_space])
                step += 1
                if done:
                    verbose = "Episodes: " + str(episodes) + ", Score: " + str(step) + '\n'
                    print(verbose)
                    self.txtbox.insert(tk.END, verbose)
                    self.txtbox.update()
                    break
                log.write(episodes, state, action, reward, step)
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
        
        mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
        play(gym.make("CartPole-v0"), fps = 15, keys_to_action = mapping, txtbox = self.txtbox)
        
        for i in self.buttons:
            i.configure(state = "normal")
        for i in self.radiobuttons:
            i.configure(state = "normal") 
        self.rb_mem.configure(state = "normal")
        



