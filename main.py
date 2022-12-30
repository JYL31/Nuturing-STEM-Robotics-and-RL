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
    root = ctk.CTk()
    plt.style.use('ggplot')
    fig = Figure(figsize=(4.5,3),dpi=100)    
    view = display(root, fig)
    view.setup()
    
    root.mainloop()