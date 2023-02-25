# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 13:51:47 2023

@author: Jiayuan Liu
"""

import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import os

class log_viewer:
    
    def __init__(self, window):
        self.window = window
        
        self.topframe = tk.LabelFrame(self.window, text="Log Data")
        self.topframe.place(height=500, width=800)
        
        self.botframe = tk.LabelFrame(self.window, text="Open File")
        self.botframe.place(height=100, width=400, rely=0.8, relx=0.25)
        
        self.load_button = tk.Button(self.botframe, text="Browse A File", command=self.load_data)
        self.load_button.place(rely=0.65, relx=0.50)
        
        self.label_file = ttk.Label(self.botframe, text="No File Selected")
        self.label_file.place(rely=0, relx=0)

        self.table = ttk.Treeview(self.topframe)
        self.table.place(relheight=1, relwidth=1)

        self.scrolly = tk.Scrollbar(self.topframe, orient="vertical", command=self.table.yview)
        self.scrollx = tk.Scrollbar(self.topframe, orient="horizontal", command=self.table.xview)
        self.table.configure(xscrollcommand=self.scrollx.set, yscrollcommand=self.scrolly.set)
        self.scrollx.pack(side="bottom", fill="x") 
        self.scrolly.pack(side="right", fill="y") 
    
    def load_data(self):
        cwd = os.getcwd()
        filename = filedialog.askopenfilename(initialdir=cwd,
                                              title="Select A File",
                                              filetype=(("csv files", "*.csv"),("All Files", "*.*")))
        self.label_file["text"] = filename
        file_path = self.label_file["text"]
        df = pd.read_csv(file_path)
        
        self.clear_data()
        self.table["column"] = list(df.columns)
        self.table["show"] = "headings"
        for column in self.table["columns"]:
            self.table.heading(column, text=column)
            
        df_rows = df.to_numpy().tolist()
        for row in df_rows:
            self.table.insert("", "end", values=row)

    def clear_data(self):
        self.table.delete(*self.table.get_children())
