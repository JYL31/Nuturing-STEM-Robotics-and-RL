# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 14:13:46 2022

@author: Jiayuan Liu
"""

import customtkinter as ctk
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from GUI import display

if __name__ == '__main__':
    
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")    
    
    root = ctk.CTk() # initialize root window
    root.title("Basic GUI Layout with Grid")
    root.maxsize(1500,  900)
    root.resizable(0, 0)
    
    plt.style.use('ggplot') # initialize figure to plot cumulative reward over episodes
    fig1 = Figure(figsize=(4.5,3),dpi=100)   
    fig2 = Figure(figsize=(4.5,3),dpi=100) 
    
    tab = ctk.CTkTabview(root)
    tab.pack()
    cartpole = tab.add("Cartpole") # tabview for each environment
    racing = tab.add("Car Racing")
    
    cartpole_view = display(cartpole, fig1, 0)
    cartpole_view.setup()
    racing_view = display(racing, fig2, 1)
    racing_view.setup()
    
    root.mainloop()