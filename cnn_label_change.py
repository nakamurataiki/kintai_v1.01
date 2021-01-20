# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:43:11 2021

@author: nakamura-taiki
"""


from tkinter import *
from tkinter import ttk

import sys
import os

import cv2



import gspread
from oauth2client.service_account import ServiceAccountCredentials

#from cnn_learn import cnn_predict

scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials.from_json_keyfile_name('C:/Users/nakamura-taiki/Google ドライブ/yolo/pacific-primer-292502-99aae112e61b.json', scope)

gc = gspread.authorize(credentials)

SPREADSHEET_KEY = '1CXVSTJzzOe0TU-0kg_qG4PZDhH8a0PXdzK_P6xrhAFU'







def label_change(predict_time, pre0, pre1, pre2):
    
    time = predict_time
    print(time)
    
    print('この関数に入りました')
    
    root = Tk()
    root.title('名前を選択してください')
    
    frame1 = ttk.Frame(root, padding=10)
    frame1.grid()
    
# Style - Theme
# Label Frame
    label_frame1 = ttk.Labelframe(frame1, text='Options', padding=(10), style='My.TLabelframe')
    label_frame1.grid(row=0, column=0)
    
    
    
    
    new_label = StringVar(value=pre0)
    radiobutton1 = ttk.Radiobutton(label_frame1, text=pre0, value='A', variable=new_label)
    radiobutton1.grid(row=0, column=0)

    
    radiobutton2 = ttk.Radiobutton(label_frame1, text=pre1, value='B', variable=new_label)
    radiobutton2.grid(row=1, column=0)
    
    radiobutton3 = ttk.Radiobutton(label_frame1, text=pre1, value='C', variable=new_label)
    radiobutton3.grid(row=2, column=0)
    
    button1 = ttk.Button(frame1, text='OK', padding=(20, 5), command=lambda : print("new_label=%s" % new_label.get()))
    button1.grid(row=3, column=0)
    
    new = new_label.get()
    print(new)
    
    new_content(new, time)
    
    
    
def new_content(new, time):
    global predict_name, timer, root, value1, value2
    
    
    print('new_contentに入りました')
    
    value1 = new
    value2 = time
    
    root = Tk()
    
    root.title('名前変更結果')

    root.resizable(False, False)
    frame1 = ttk.Frame(root, padding=(64))
    frame1.grid()
    
    predict_name = StringVar(value=value1)
    predict_name_entry = ttk.Entry(frame1, textvariable=predict_name, width=20)
    predict_name_entry.grid(row=1, column=2)
    
    timer = StringVar(value = value2)
    timer_entry = ttk.Entry(frame1, textvariable=timer, width=20)
    timer_entry.grid(row=2, column=2)
    
    frame2 = ttk.Frame(frame1, padding=(0, 0))
    frame2.grid(row=3, column=1)
    
    button1 = ttk.Button(frame2, text='出社', command=btn_in_click)
    button1.pack(side=LEFT, pady=10)
    
    frame3 = ttk.Frame(frame1, padding=(0, 0))
    frame3.grid(row=3, column=2)
    
    button2 = ttk.Button(frame3, text='退社', command=btn_out_click)
    button2.pack(side=LEFT, pady=10)
    
    frame4 = ttk.Frame(frame1, padding=(0, 0))
    frame4.grid(row=3, column=3)
    
    button3 = ttk.Button(frame4, text='休み', command=btn_off_click)
    button3.pack(side=LEFT, pady=10)
    

    
    
    
    
    
    
def btn_in_click():
    global predict_name
    print('出社押された')
    
    
    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1
    print('OK')
    if value1 == 'tukamoto':
        worksheet.update_cell(13, 3, '出社')
        worksheet.update_cell(13, 4, value2)
    elif value1 == 'nakamura':
        worksheet.update_cell(16, 3, '出社')
        worksheet.update_cell(16, 4, value2)
    elif value1 == 'kobayakawa':                        
        worksheet.update_cell(14, 3, '出社')
        worksheet.update_cell(14, 4, value2)
    
    
    
    
    

    predict_name.set("")
    timer.set("")


def btn_out_click():
    print('退社押された')
    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1

    if value1 == 'tukamoto':
        worksheet.update_cell(13, 3, '')
        worksheet.update_cell(13, 5, value2)
    elif value1 == 'nakamura':
        worksheet.update_cell(16, 3, '')
        worksheet.update_cell(16, 5, value2)
    elif value1 == 'kobayakawa':                        
        worksheet.update_cell(14, 3, '')
        worksheet.update_cell(14, 5, value2)




    predict_name.set("")
    timer.set("")
    
    
def btn_off_click():
    print('休み押された')

    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1
    if value1 == 'tukamoto':
        worksheet.update_cell(13, 3, '休み')
    elif value1 == 'nakamura':
        worksheet.update_cell(16, 3, '休み')
    elif value1 == 'kobayakawa':                        
        worksheet.update_cell(14, 3, '休み')



    predict_name.set("")
    timer.set("")
    
    
    