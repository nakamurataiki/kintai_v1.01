# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:29:59 2021

@author: nakamura-taiki
"""


from tkinter import *
#from tkinter import ttk
from tkinter import font

import tkinter as tk
import tkinter.ttk as ttk



def select(value):
    global var_option
    print(value)       #ここで値が受け渡される
    #print(var_option.get())



    
root = tk.Tk()

root.title('検出結果')

root.resizable(False, False)
frame1 = ttk.Frame(root, padding=(64))
frame1.grid()


lst = ["Value A", "Value B", "Value C"]

face_image = PhotoImage(file='C:/Users/nakamura-taiki/.conda/envs/yolo_test2/yolo/Image_0_138.png', master=root)

label1 = ttk.Label(frame1, image=face_image, compound=TOP, padding=(10, 5, 10, 30))
label1.grid(row=0, column=2)


frame2 = ttk.Frame(frame1, padding=(0, 0))
frame2.grid(column=2)



var_option = tk.StringVar(value="Value A")
print(var_option.get())
optionmenu = tk.OptionMenu(frame2, var_option, *lst, command=select)
optionmenu.pack(padx=10, pady=10)




root.mainloop()

