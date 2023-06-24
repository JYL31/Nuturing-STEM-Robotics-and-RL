# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:53:18 2022

@author: Jiayuan Liu
"""

import cv2
import pygame
import random
import tkinter as tk
import customtkinter as ctk
import numpy as np
import pandas as pd
from keras.models import load_model
from collections import deque
from cartpole import CartPoleEnv
from custom_racing import CarRacing
from util_play import play
from Log_Veiwer import log_viewer
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

class display:
    
    def __init__(self, root, fig, tab):
        self.container = root
        self.fig = fig
        self.model = None
        self.tab = tab

    def setup(self):
        self.create_widgets()
        self.setup_layout()  

    # initialize buttons and plots
    def create_widgets(self):
        self.leftFrame = ctk.CTkFrame(self.container, width=100, height=780)
        self.midFrame = ctk.CTkFrame(self.container, width=300, height=780)
        self.rightFrame = ctk.CTkFrame(self.container, width=400, height=800)
        self.rightFrame1 = ctk.CTkFrame(self.rightFrame, width=400, height=380)
        self.rightFrame2 = ctk.CTkFrame(self.rightFrame, width=400, height=380)
        
        self.label_lr = ctk.CTkLabel(self.leftFrame, text='Learning rate : ')
        self.var_lr = tk.DoubleVar(value=0.001)
        self.rb_lr1 = ctk.CTkRadioButton(self.leftFrame, text='0.001', variable=self.var_lr, value=0.001)
        self.rb_lr2 = ctk.CTkRadioButton(self.leftFrame, text='0.1', variable=self.var_lr, value=0.1)
        
        if self.tab == 0:
            self.label_r = ctk.CTkLabel(self.leftFrame, text='Reward engineering : ')
            self.var_r = tk.IntVar(value=1)
            self.rb_r1 = ctk.CTkRadioButton(self.leftFrame, text='angle difference', variable=self.var_r, value=1)
            self.rb_r2 = ctk.CTkRadioButton(self.leftFrame, text='+1 each step', variable=self.var_r, value=0)
        
        self.label_fps = ctk.CTkLabel(self.leftFrame, text='FPS : ')
        self.var_fps = tk.DoubleVar(value=15)
        self.rb_fps1 = ctk.CTkRadioButton(self.leftFrame, text='15', variable=self.var_fps, value=15)
        self.rb_fps2 = ctk.CTkRadioButton(self.leftFrame, text='50', variable=self.var_fps, value=50)
        
        self.but_train = ctk.CTkButton(self.leftFrame, text='Start Training', command=self.train_network)
        self.but_test = ctk.CTkButton(self.leftFrame, text='Start Testing', command=self.test_network, state="disabled")
        self.but_reset = ctk.CTkButton(self.leftFrame, text='Reset to Default', command=self.reset)
        self.but_load = ctk.CTkButton(self.leftFrame, text='Load Network Model', command=self.load)
        self.but_log = ctk.CTkButton(self.leftFrame, text='Log', command=self.open_csv)
        self.but_play = ctk.CTkButton(self.leftFrame, text='Play', command=self.manual_play)
        self.but_replay = ctk.CTkButton(self.leftFrame, text='Replay', command=self.replay, state="disabled")
        self.entry = ctk.CTkEntry(self.leftFrame, placeholder_text="1 ~ 100")
        
        self.label_nn = ctk.CTkLabel(self.midFrame, text='Deep Q Network Structure')
        self.canvas_nn = ctk.CTkCanvas(self.midFrame, width=340, height=700)
    
        self.label_plot = ctk.CTkLabel(self.rightFrame1, text='Reward over episodes')
        self.canvas_plot = FigureCanvasTkAgg(self.fig, self.rightFrame1)
        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, self.rightFrame1)
        
        self.label_txt = ctk.CTkLabel(self.rightFrame2, text='Verbose')
        self.txtbox = ctk.CTkTextbox(self.rightFrame2, width=400, height=390)
        if self.tab == 0:
            self.radiobuttons = [self.rb_lr1, self.rb_lr2, self.rb_r1, self.rb_r2,
                                 self.rb_fps1, self.rb_fps2]
        else:
            self.radiobuttons = [self.rb_lr1, self.rb_lr2, self.rb_fps1, self.rb_fps2]
        
        self.buttons = [self.but_train, self.but_test, self.but_reset,
                        self.but_load, self.but_log, self.but_play, self.but_replay]

    # setup the layout for buttons and plots    
    def setup_layout(self):
        self.leftFrame.grid(row=0, column=0, padx=10, pady=5)
        self.midFrame.grid(row=0, column=1, padx=10, pady=5)
        self.rightFrame.grid(row=0, column=2, padx=10, pady=5)
        self.rightFrame.grid_propagate(False)
        
        self.label_fps.grid(row=0, column=0, padx=5, pady=5)
        self.rb_fps1.grid(row=0, column=1, padx=5, pady=5)
        self.rb_fps2.grid(row=0, column=2, padx=5, pady=5)
        
        self.label_lr.grid(row=1, column=0, padx=5, pady=5)
        self.rb_lr1.grid(row=1, column=1, padx=5, pady=5)
        self.rb_lr2.grid(row=1, column=2, padx=5, pady=5)
        
        if self.tab == 0:
            self.label_r.grid(row=2, column=0, padx=5, pady=5)
            self.rb_r1.grid(row=2, column=1, padx=5, pady=5)
            self.rb_r2.grid(row=2, column=2, padx=5, pady=5)
        
        self.but_train.grid(row=3, column=1, padx=5, pady=5)
        self.but_test.grid(row=4, column=1, padx=5, pady=5)
        self.but_reset.grid(row=5, column=1, padx=5, pady=5)
        self.but_load.grid(row=6, column=1, padx=5, pady=5)
        self.but_log.grid(row=7, column=1, padx=5, pady=5)
        self.but_play.grid(row=8, column=1, padx=5, pady=5)
        self.but_replay.grid(row=9, column=1, padx=5, pady=5)
        self.entry.grid(row=10, column=1, padx=5, pady=5)
        
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

    # function to load a previously trained agent                    
    def load(self):
        
        # load network model and plot corresponding to the selected training parameters
        if self.tab == 0:
            data_no = random.randrange(0,3)
            name = "r{}_lr{}_{}".format(self.var_r.get(), self.var_lr.get(), data_no)
            self.model = load_model("Cartpole\{}".format(name))
            self.model_plot = tk.PhotoImage(file = r'Cartpole\model_plot.png')
        else:
            name = "lr{}".format(self.var_lr.get())
            self.model = load_model("CarRacing\{}".format(name))
            self.model_plot = tk.PhotoImage(file = r'CarRacing\model_plot.png')
        
        reward_type = "angle difference" if self.var_r.get()==1 else "+1 each step"
        
        # print out the loading information
        verbose = "Loaded model with training parameters: \n\
            --Reward Engineering: {}\n\
            --Learning rate: {}\n".format(reward_type, self.var_lr.get())
        print(verbose)
        self.txtbox.insert(tk.END, verbose)
        self.txtbox.update()
        
        self.canvas_nn.create_image(172, 265, image=self.model_plot)
        self.canvas_nn.update()
        
        # set buttons' states
        self.but_train.configure(state="disabled")
        for i in self.radiobuttons:
            i.configure(state="disabled")
        self.but_test.configure(state="normal")

    # function to delete an agent, reset buttons, and clear out plots and texts
    def reset(self):
        self.var_lr.set(value=0.001)
        if self.tab == 0:
            self.var_r.set(value=1)
        self.var_fps.set(15)

        self.canvas_nn.delete("all")
        if hasattr(self, 'ax'):
            self.ax.clear()
            self.canvas_plot.draw()
        self.txtbox.delete("1.0", "end")
        self.entry.delete("0", "end")
        
        for i in self.buttons:
            i.configure(state="normal")
        for i in self.radiobuttons:
            i.configure(state="normal")
        self.but_test.configure(state="disabled")
        self.but_replay.configure(state="disabled")
        
        self.model = None
    
    # function to update the cumulative reward over episode plot in real time
    def animate(self):
        self.ax.clear()
        self.ax.plot(self.rewards)
        plt.pause(0.5)
        self.canvas_plot.draw()

    # function to train the network
    def train_network(self):
        
        # disable all interactables when training is in progress
        for i in self.buttons:
            i.configure(state="disabled")
        for i in self.radiobuttons:
            i.configure(state="disabled")
        
        # delete previous plots and texts
        self.canvas_nn.delete("all")
        if hasattr(self, 'ax'):
            self.ax.clear()
            self.canvas_plot.draw()
        self.txtbox.delete("1.0", "end")
        
        self.rewards = []
        self.ax = self.fig.add_subplot(111)
        
        env_type = self.tab
        data_no = random.randrange(0,3)

        # initialize the environment and an agent with the selected training parameter
        if env_type == 0:
            name = "r{}_lr{}_{}".format(self.var_r.get(), self.var_lr.get(), data_no)
            data_dir = "Cartpole/Logs/{}.csv".format(name)
            self.model = load_model("Cartpole\{}".format(name))
            self.model_plot = tk.PhotoImage(file = r'Cartpole\model_plot.png')
            env = CartPoleEnv(render_mode='human')
            self.para = [self.var_r.get(), self.var_lr.get(), data_no]
        else:
            name = "lr{}".format(self.var_lr.get())
            data_dir = "CarRacing/Logs/{}.csv".format(name)
            self.model = load_model("CarRacing\{}".format(name))
            self.model_plot = tk.PhotoImage(file = r'CarRacing\model_plot.png')
            env = CarRacing(continuous=True, render_mode='human')
            action_space = [(0, 1, 0), (-1, 0, 0), (0, 0, 0), #           Action Space Structure
                            (1, 0, 0), (0, 0, 0.5)]           #        (Steering Wheel, Gas, Break)
                                                              # Range        -1~1       0~1   0~1
            self.para = [self.var_lr.get(), data_no]
        
        # get the dataset of pre-recorded training
        data = pd.read_csv(data_dir)
        
        self.canvas_nn.create_image(172, 265, image=self.model_plot)
        self.canvas_nn.update()
        
        for episode in range(1, 101):
            # reset the environment to a fixed seed so that pre-recorded dataset can be used to replay training
            if env_type == 0:
                env.reset(seed=5)
            else:
                env.reset(seed=10)
            episode_data = data.loc[data["Episode Number"]==episode]
            i = 0
            terminal = False
            while True:
                if episode % 5 == 0:
                    env.render()
                    action = episode_data["Action"].iloc[i]
                    if env_type == 1:
                        action = action_space[action]
                        for j in range(3): # skip 3 frames
                            _, _, terminal, _, _ = env.step(action)
                            if terminal:
                                break
                    else:
                        _, _, terminal, _, _ = env.step(action)
                score = episode_data["Score"].iloc[i]
                if terminal or episode % 5 != 0 or score < 0:
                    score = episode_data["Score"].iloc[-1]
                    exploration_rate = episode_data["Exploration Rate"].iloc[-1]
                    verbose = "Episodes: {0}, Exploration: {1:.4f}, Score: {2:.2f}\n".format(episode, exploration_rate, score)
                    print(verbose)
                    self.txtbox.insert(tk.END, verbose)
                    self.txtbox.update()
                    self.txtbox.see("end")
                    self.rewards.append(score)
                    self.animate()
                    break
                i += 1
        verbose = "Training Completed!\n"
        print(verbose)
        self.txtbox.insert(tk.END, verbose)
        self.txtbox.update()

        # reset buttons' states
        env.close()
        for i in self.buttons:
            if i != self.but_train:
                i.configure(state="normal")
        for i in self.radiobuttons:
            i.configure(state="normal")

    # function to test the trained agent
    def test_network(self):
        for i in self.buttons:
            i.configure(state="disabled")
        for i in self.radiobuttons:
            i.configure(state="disabled")
        
        self.canvas_nn.delete("all")
        if hasattr(self, 'ax'):
            self.ax.clear()
            self.canvas_plot.draw()
        self.txtbox.delete("1.0", "end")
        
        env_type = self.tab
            
        if env_type == 0:
            env = CartPoleEnv(render_mode='human')
            observation_space = env.observation_space.shape[0]
            state = env.reset(seed=5)
            state = np.reshape(state, [1, observation_space])
        else:
            env = CarRacing(continuous=True, render_mode='human')
            action_space = [(0, 1, 0), (-1, 0, 0), (0, 0, 0), #           Action Space Structure
                            (1, 0, 0), (0, 0, 0.5)]           #        (Steering Wheel, Gas, Break)
                                                              # Range        -1~1       0~1   0~1
            state = env.reset(seed=10)
            state = cv2.cvtColor(state[0], cv2.COLOR_BGR2GRAY)
            state = state.astype(float)
            state /= 255.0
            frame_stack = deque([state]*5, maxlen=5) # 5 stacks of image as state
            
        self.canvas_nn.create_image(172, 265, image=self.model_plot)
        self.canvas_nn.update()
        
        score = 0  
        while True:
            env.render()
            if env_type == 0: # Cartpole
                action = np.argmax(self.model.predict(state, verbose=0))
                state_next, reward, terminal, info, _ = env.step(action)
                reward = -100*(abs(state_next[2]) - abs(state[0][2]))
                state = np.reshape(state_next, [1, observation_space])
                score += 1
            else: # Car Racing
                cur_frame_stack = np.array(frame_stack)
                cur_frame_stack = np.transpose(cur_frame_stack, (1, 2, 0))
                cur_frame_stack = np.expand_dims(cur_frame_stack, axis=0)
                action = np.argmax(self.model.predict(cur_frame_stack, verbose=0))
            
                reward = 0
                for i in range(3): # skip 3 frames
                    state_next, r, terminal, info, _ = env.step(action_space[action])
                    reward += r
                    if terminal:
                        break
                state = state_next[1]
                state_next = state_next[0]
    
                if action_space[action][1] == 1 and reward > 0:
                    reward = 1.2*reward
                score += reward
        
                state_next = cv2.cvtColor(state_next, cv2.COLOR_BGR2GRAY)
                state_next = state_next.astype(float)
                state_next /= 255.0
                frame_stack.append(state_next)
                    
            if terminal:
                verbose = "Score: " + str(score) + '\n'
                print(verbose)
                self.txtbox.insert(tk.END, verbose)
                self.txtbox.update()
                break
        env.close()
        for i in self.buttons:
            i.configure(state="normal")
        for i in self.radiobuttons:
            i.configure(state="normal") 
        
        if hasattr(self, 'para')==False:
            self.but_replay.configure(state="disabled")

    # fucntion to open up a new window to visualize csv file content
    def open_csv(self):
        
        def destroy():
            self.table_window.destroy()
        
        self.table_window = ctk.CTkToplevel(self.container)
        self.table_window.protocol("WM_DELETE_WINDOW", destroy)
        self.table_window.geometry("800x650")
        self.table_window.pack_propagate(False)
        self.table_window.resizable(0, 0)
        self.table_GUI = log_viewer(self.table_window)
    
    # function to play the environment using keyboard
    def manual_play(self):
        for i in self.buttons:
            i.configure(state="disabled")
        for i in self.radiobuttons:
            i.configure(state="disabled")
        
        self.canvas_nn.delete("all")
        if hasattr(self, 'ax'):
            self.ax.clear()
            self.canvas_plot.draw()
        self.txtbox.delete("1.0", "end")
        
        verbose = "--Press SPACE to start the game\n--ARROW keys to control\n--Press ESC to exit the game\n"
        print(verbose)
        self.txtbox.insert(tk.END, verbose)
        self.txtbox.update()
        
        env_type = self.tab
        
        if env_type == 0:
            mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
            play(CartPoleEnv(play=True), fps = self.var_fps.get(), keys_to_action = mapping, txtbox = self.txtbox)
        else:
            mapping = {(pygame.K_LEFT,): [-1, 0, 0], (pygame.K_RIGHT,): [1, 0, 0],
                       (pygame.K_UP,): [0, 1, 0], (pygame.K_DOWN,): [0, 0, 0.5]}
            play(CarRacing(continuous=True), fps = self.var_fps.get(), keys_to_action = mapping,
                 txtbox = self.txtbox, seed=10, continuous=True)
        
        for i in self.buttons:
            i.configure(state = "normal")
        for i in self.radiobuttons:
            i.configure(state = "normal") 
        if self.model == None:
            self.but_test.configure(state="disabled")
            self.but_replay.configure(state="disabled")

    # function to replay a previous training episode specified by the entry box    
    def replay(self):
        for i in self.buttons:
            i.configure(state="disabled")
        for i in self.radiobuttons:
            i.configure(state="disabled")
        
        self.canvas_nn.delete("all")
        if hasattr(self, 'ax'):
            self.ax.clear()
            self.canvas_plot.draw()
        self.txtbox.delete("1.0", "end")
        
        try:
            episode = int(self.entry.get())
            if episode < 1 or episode > 100:
                raise ValueError()
        except (ValueError):
            verbose = "Please enter an integer between 1 and 100"
            print(verbose)
            self.txtbox.insert(tk.END, verbose)
            self.txtbox.update()
            for i in self.buttons:
                i.configure(state="normal")
            for i in self.radiobuttons:
                i.configure(state="normal")
            return
        
        env_type = self.tab
        
        if env_type == 0:
            env = CartPoleEnv(render_mode='human')
            env.reset(seed=5)
            data_dir = "Cartpole/Logs/r{}_lr{}_{}.csv".format(self.para[0], self.para[1], self.para[2])
            data = pd.read_csv(data_dir)
        else:
            env = CarRacing(continuous=True, render_mode='human')
            action_space = [(0, 1, 0), (-1, 0, 0), (0, 0, 0), #           Action Space Structure
                            (1, 0, 0), (0, 0, 0.5)]           #        (Steering Wheel, Gas, Break)
                                                              # Range        -1~1       0~1   0~1
            env.reset(seed=10)
            data_dir = "CarRacing/Logs/lr{}.csv".format(self.para[0])
        
        data = pd.read_csv(data_dir)
        if env_type == 0:
            reward_type = "angle difference" if self.para[0]==1 else "+1 each step"
        else:
            reward_type = "number of tiles passed"
        verbose = "Replaying Episode {} from last training instance with settings: \n\
            --Reward Engineering: {}\n\
            --Learning rate: {}\n".format(episode, reward_type, self.para[1])
        print(verbose)
        self.txtbox.insert(tk.END, verbose)
        self.txtbox.update()
        
        self.canvas_nn.create_image(172, 265, image=self.model_plot)
        self.canvas_nn.update()
        
        episode_data = data.loc[data["Episode Number"]==episode]
        i = 0
        while True:
            if env_type == 0:
                env.render(episode=episode)
            else:
                env.render()
            action = episode_data["Action"].iloc[i]
            if env_type == 1:
                action = action_space[action]
                for j in range(3):
                    _, _, terminal, _, _ = env.step(action)
                    if terminal:
                        break
            else:
                _, _, terminal, _, _ = env.step(action)
            if terminal:
                score = episode_data["Score"].iloc[-1]
                exploration_rate = episode_data["Exploration Rate"].iloc[-1]
                verbose = "Episodes: {0}, Exploration: {1:.4f}, Score: {2:.2f}\n".format(episode, exploration_rate, score)
                print(verbose)
                self.txtbox.insert(tk.END, verbose)
                self.txtbox.update()
                break
            i += 1
        verbose = "Replay finished!\n"
        print(verbose)
        self.txtbox.insert(tk.END, verbose)
        self.txtbox.update()
        
        env.close()
        for i in self.buttons:
            i.configure(state="normal")
        for i in self.radiobuttons:
            i.configure(state="normal")
        
        



