# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 11:53:18 2022

@author: Jiayuan Liu
"""

import gym
import tkinter as tk
import customtkinter as ctk
import numpy as np
from statistics import mean
from Agent import DQN_Agent
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
        self.rightFrame1 = ctk.CTkFrame(self.rightFrame, width=400, height=390)
        self.rightFrame2 = ctk.CTkFrame(self.rightFrame, width=400, height=390)
        
        self.label_exr = ctk.CTkLabel(self.leftFrame, text='exploration rate : ')
        self.var_exr = tk.DoubleVar(value=1)
        self.rb_exr1 = ctk.CTkRadioButton(self.leftFrame, text='1', variable=self.var_exr, value=1)
        self.rb_exr2 = ctk.CTkRadioButton(self.leftFrame, text='0.5', variable=self.var_exr, value=0.5)
        self.rb_exr3 = ctk.CTkRadioButton(self.leftFrame, text='0.1', variable=self.var_exr, value=0.1)

        self.label_exd = ctk.CTkLabel(self.leftFrame, text='exploration decay : ')
        self.var_exd = tk.DoubleVar(value=0.999)
        self.rb_exd1 = ctk.CTkRadioButton(self.leftFrame, text='0.999', variable=self.var_exd, value=0.999)
        self.rb_exd2 = ctk.CTkRadioButton(self.leftFrame, text='0.5', variable=self.var_exd, value=0.5)
        self.rb_exd3 = ctk.CTkRadioButton(self.leftFrame, text='0.1', variable=self.var_exd, value=0.1)
        
        self.label_lr = ctk.CTkLabel(self.leftFrame, text='learning rate : ')
        self.var_lr = tk.DoubleVar(value=0.001)
        self.rb_lr1 = ctk.CTkRadioButton(self.leftFrame, text='0.001', variable=self.var_lr, value=0.001)
        self.rb_lr2 = ctk.CTkRadioButton(self.leftFrame, text='0.01', variable=self.var_lr, value=0.01)
        self.rb_lr3 = ctk.CTkRadioButton(self.leftFrame, text='0.1', variable=self.var_lr, value=0.1)
        
        self.label_df = ctk.CTkLabel(self.leftFrame, text='discount factor : ')
        self.var_df = tk.DoubleVar(value=0.95)
        self.rb_df1 = ctk.CTkRadioButton(self.leftFrame, text='0.95', variable=self.var_df, value=0.95)
        self.rb_df2 = ctk.CTkRadioButton(self.leftFrame, text='0.5', variable=self.var_df, value=0.5)
        self.rb_df3 = ctk.CTkRadioButton(self.leftFrame, text='0.1', variable=self.var_df, value=0.1)
        
        self.label_mem = ctk.CTkLabel(self.leftFrame, text='memory size : ')
        self.var_mem = tk.IntVar(value=2000)
        self.rb_mem1 = ctk.CTkRadioButton(self.leftFrame, text='2000', variable=self.var_mem, value=2000)
        self.rb_mem2 = ctk.CTkRadioButton(self.leftFrame, text='200', variable=self.var_mem, value=200)
        
        self.label_lyr1 = ctk.CTkLabel(self.leftFrame, text='layer 1 units : ')
        self.var_lyr1 = tk.IntVar(value=64)
        self.rb_lyr11 = ctk.CTkRadioButton(self.leftFrame, text='64', variable=self.var_lyr1, value=64)
        self.rb_lyr12 = ctk.CTkRadioButton(self.leftFrame, text='32', variable=self.var_lyr1, value=32)
        self.rb_lyr13 = ctk.CTkRadioButton(self.leftFrame, text='16', variable=self.var_lyr1, value=16)
        
        self.label_lyr2 = ctk.CTkLabel(self.leftFrame, text='layer 2 units : ')
        self.var_lyr2 = tk.IntVar(value=32)
        self.rb_lyr21 = ctk.CTkRadioButton(self.leftFrame, text='64', variable=self.var_lyr2, value=64)
        self.rb_lyr22 = ctk.CTkRadioButton(self.leftFrame, text='32', variable=self.var_lyr2, value=32)
        self.rb_lyr23 = ctk.CTkRadioButton(self.leftFrame, text='16', variable=self.var_lyr2, value=16)
        
        self.label_lyr3 = ctk.CTkLabel(self.leftFrame, text='layer 3 units : ')
        self.var_lyr3 = tk.IntVar(value=16)
        self.rb_lyr31 = ctk.CTkRadioButton(self.leftFrame, text='64', variable=self.var_lyr3, value=64)
        self.rb_lyr32 = ctk.CTkRadioButton(self.leftFrame, text='32', variable=self.var_lyr3, value=32)
        self.rb_lyr33 = ctk.CTkRadioButton(self.leftFrame, text='16', variable=self.var_lyr3, value=16)
        
        self.but_train = ctk.CTkButton(self.leftFrame, text='Start Training', command=self.train_network)
        self.but_test = ctk.CTkButton(self.leftFrame, text='Start Testing', command=self.test_network)
        self.but_reset = ctk.CTkButton(self.leftFrame, text='Reset to Default', command=self.reset)
        self.but_save = ctk.CTkButton(self.leftFrame, text='Save Network Model', command=self.save)
        self.but_load = ctk.CTkButton(self.leftFrame, text='Load Network Model', command=self.load)
        
        self.label_nn = ctk.CTkLabel(self.midFrame, text='Deep Q Network Structure')
        self.canvas_nn = ctk.CTkCanvas(self.midFrame, width=341, height=526)
    
        self.label_plot = ctk.CTkLabel(self.rightFrame1, text='Reward over episodes')
        self.canvas_plot = FigureCanvasTkAgg(self.fig, self.rightFrame1)
        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, self.rightFrame1)
        
        self.label_txt = ctk.CTkLabel(self.rightFrame2, text='Training Verbose')
        self.txtbox = ctk.CTkTextbox(self.rightFrame2, width=400, height=390)
        
        self.radiobuttons = [self.rb_exr1, self.rb_exr2, self.rb_exr3,
                             self.rb_exd1, self.rb_exd2, self.rb_exd3, 
                             self.rb_lr1, self.rb_lr2, self.rb_lr3, 
                             self.rb_df1, self.rb_df2, self.rb_df3, 
                             self.rb_mem1, self.rb_mem2, self.rb_lyr11, 
                             self.rb_lyr12, self.rb_lyr13, self.rb_lyr21,
                             self.rb_lyr22, self.rb_lyr23, self.rb_lyr31, 
                             self.rb_lyr32, self.rb_lyr33]
        
        self.buttons = [self.but_train, self.but_test, self.but_reset,
                        self.but_save, self.but_load]
        
    def setup_layout(self):
        self.leftFrame.grid(row=0, column=0, padx=10, pady=5)
        self.midFrame.grid(row=0, column=1, padx=10, pady=5)
        self.rightFrame.grid(row=0, column=2, padx=10, pady=5)
        
        self.label_exr.grid(row=0, column=0, padx=5, pady=5)
        self.rb_exr1.grid(row=0, column=1, padx=5, pady=5)
        self.rb_exr2.grid(row=0, column=2, padx=5, pady=5)
        self.rb_exr3.grid(row=0, column=3, padx=5, pady=5)
        
        self.label_exd.grid(row=1, column=0, padx=5, pady=5)
        self.rb_exd1.grid(row=1, column=1, padx=5, pady=5)
        self.rb_exd2.grid(row=1, column=2, padx=5, pady=5)
        self.rb_exd3.grid(row=1, column=3, padx=5, pady=5)
        
        self.label_lr.grid(row=2, column=0, padx=5, pady=5)
        self.rb_lr1.grid(row=2, column=1, padx=5, pady=5)
        self.rb_lr2.grid(row=2, column=2, padx=5, pady=5)
        self.rb_lr3.grid(row=2, column=3, padx=5, pady=5)
        
        self.label_df.grid(row=3, column=0, padx=5, pady=5)
        self.rb_df1.grid(row=3, column=1, padx=5, pady=5)
        self.rb_df2.grid(row=3, column=2, padx=5, pady=5)
        self.rb_df3.grid(row=3, column=3, padx=5, pady=5)
        
        self.label_mem.grid(row=4, column=0, padx=5, pady=5)
        self.rb_mem1.grid(row=4, column=1, padx=5, pady=5)
        self.rb_mem2.grid(row=4, column=2, padx=5, pady=5)
        
        self.label_lyr1.grid(row=5, column=0, padx=5, pady=5)
        self.rb_lyr11.grid(row=5, column=1, padx=5, pady=5)
        self.rb_lyr12.grid(row=5, column=2, padx=5, pady=5)
        self.rb_lyr13.grid(row=5, column=3, padx=5, pady=5)
        
        self.label_lyr2.grid(row=6, column=0, padx=5, pady=5)
        self.rb_lyr21.grid(row=6, column=1, padx=5, pady=5)
        self.rb_lyr22.grid(row=6, column=2, padx=5, pady=5)
        self.rb_lyr23.grid(row=6, column=3, padx=5, pady=5)
        
        self.label_lyr3.grid(row=7, column=0, padx=5, pady=5)
        self.rb_lyr31.grid(row=7, column=1, padx=5, pady=5)
        self.rb_lyr32.grid(row=7, column=2, padx=5, pady=5)
        self.rb_lyr33.grid(row=7, column=3, padx=5, pady=5)
        
        self.but_train.grid(row=8, column=1, padx=5, pady=5)
        self.but_test.grid(row=9, column=1, padx=5, pady=5)
        self.but_reset.grid(row=10, column=1, padx=5, pady=5)
        self.but_save.grid(row=11, column=1, padx=5, pady=5)
        self.but_load.grid(row=12, column=1, padx=5, pady=5)
        
        self.label_nn.grid(row=0, column=0, padx=5, pady=5)
        self.canvas_nn.grid(row=1, column=0, padx=5, pady=5)
        
        self.rightFrame1.pack(side='top')
        self.rightFrame2.pack(side='bottom')
        
        self.label_plot.pack(side='top')
        self.canvas_plot.get_tk_widget().pack(side='bottom')
        self.toolbar.pack(side='bottom')
        
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
        self.var_exd.set(value=0.999)
        self.var_lr.set(value=0.001)
        self.var_df.set(value=0.95)
        self.var_mem.set(value=2000)
        self.var_lyr1.set(value=64)
        self.var_lyr2.set(value=32)
        self.var_lyr3.set(value=16)
        self.canvas_nn.delete("all")
        self.canvas_plot.get_tk_widget().delete("all")
        self.txtbox.delete("1.0", "end")
        
        for i in self.buttons:
            i.configure(state="normal")
        for i in self.radiobuttons:
            i.configure(state="normal")
        
        if isinstance(self.agent, DQN_Agent):
            del self.agent
    
    def animate(self,i):
        self.ax.clear()
        self.ax.plot(self.rewards)
        plt.pause(1)

    
    def train_network(self):
        for i in self.buttons:
            i.configure(state="disabled")
        for i in self.radiobuttons:
            i.configure(state="disabled")    
        
        self.rewards = []
        self.ax = self.fig.add_subplot(111)
        env = gym.make('CartPole-v0') #, render_mode='human')
        observation_space = env.observation_space.shape[0]
        action_space = env.action_space.n
        self.layer_units = [self.var_lyr1.get(), self.var_lyr2.get(), self.var_lyr3.get()]
        self.agent = DQN_Agent(observation_space, action_space, exploration_rate=self.var_exr.get(), 
                               exploration_decay=self.var_exd.get(), learning_rate=self.var_lr.get(), 
                               discount_factor=self.var_df.get(), memory_size=self.var_mem.get(), 
                               layer_units=self.layer_units)
        print(isinstance(self.agent, DQN_Agent))
        self.image = tk.PhotoImage(file = 'model_plot.png')
        self.canvas_nn.create_image(172, 265, image=self.image)
        self.canvas_nn.update()
        ani = animation.FuncAnimation(self.fig, self.animate, interval=1000, repeat=False)
        self.canvas_plot.draw()
        run = 0
        done = False
        while run < 5 and done == False:
            run += 1
            state = env.reset()
            state = np.reshape(state, [1, observation_space])
            step = 0
            while True:
                #env.render()  
                action = self.agent.policy(state)
                state_next, reward, terminal, info = env.step(action)
                reward = reward if not terminal else -reward
                state_next = np.reshape(state_next, [1, observation_space])
                self.agent.remember(state, action, reward, state_next, terminal)
                state = state_next
                if terminal:
                    verbose = "Episodes: " + str(run) + ", Exploration: " + str(self.agent.exploration_rate) + ", Reward: " + str(step) + '\n'
                    print(verbose)
                    self.txtbox.insert(tk.END, verbose)
                    self.txtbox.update()
                    self.rewards.append(step)
                    if run >= 5 and mean(self.rewards[-4:]) >= 195:
                        self.agent.save('DQN_model')
                        done = True
                        print("Training Completed!")
                    break
                step += 1
                self.agent.experience_replay()
        env.close()
        ani._stop()
        for i in self.buttons:
            i.configure(state="normal")
        for i in self.radiobuttons:
            i.configure(state="normal") 

    def test_network(self):
        for i in self.buttons:
            i.configure(state="disabled")
        for i in self.radiobuttons:
            i.configure(state="disabled") 
        
        model = self.agent.get_model()
        env = gym.make('CartPole-v0')#, render_mode='human')
        observation_space = env.observation_space.shape[0]
        for episodes in 5:
            state = env.reset()
            state = np.reshape(state, [1, observation_space])
            done = False
            step = 0
            while not done:
                #env.render()
                action = np.argmax(model.predict(state))
                next_state, reward, done, _ = env.step(action)
                state = np.reshape(next_state, [1, observation_space])
                if done:
                    print("Episode: {}, Reward: {} \n".format(episodes, step))
                    break
                step += 1
        
        env.close()
        for i in self.buttons:
            i.configure(state="normal")
        for i in self.radiobuttons:
            i.configure(state="normal") 
        


