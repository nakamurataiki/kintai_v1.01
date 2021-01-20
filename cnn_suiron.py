# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 11:14:51 2021

@author: nakamura-taiki
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import  load_model
import sys
import os

import cnn_learn

from PIL import Image

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

from cnn_learn import cnn_learning, cnn_evaluate
from cnn_label_change import label_change

from tkinter import *
from tkinter import ttk
import tkinter as tk

import datetime

import math

import gspread
from oauth2client.service_account import ServiceAccountCredentials

import sqlite3

#from cnn_learn import cnn_predict

scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']

credentials = ServiceAccountCredentials.from_json_keyfile_name('C:/Users/nakamura-taiki/Google ドライブ/yolo/pacific-primer-292502-99aae112e61b.json', scope)

gc = gspread.authorize(credentials)

SPREADSHEET_KEY = '1CXVSTJzzOe0TU-0kg_qG4PZDhH8a0PXdzK_P6xrhAFU'






def image_cut(cascade_path):
    #DATA_DIR = "C:/work/face/new/face_new/evaluate/"    #画像と検出位置の保存ディレクトリ   パスに日本語が入っているとファイル名が文字化けする
    DATA_DIR = "C:/work/face/new/face_new/evaluate/"
#CASCADE_PATH = "C:/work/opencv/data/haarcascades/haarcascade_frontalface_default.xml"     #cv2のカスケードファイル

    #CASCADE_PATH = "C:/work/opencv/data/haarcascades/haarcascade_frontalface_alt.xml"   #カスケードファイルはaltの方が枠が出ないケースが少ないらしい
    
    #CAP_COUNT = 400      #画像取得回数  ここを何枚にするかを検討中(2020/12/11 段階)
    
    #カメラ設定
    cam = cv2.VideoCapture(0)    
    print("CamSize:", cam.get(cv2.CAP_PROP_FRAME_WIDTH),cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_w = 32*20   #取得画像の幅（畳み込み層の都合上、2^5=32の倍数)  640
    cam_h = 32*15   #取得画像の高さ                               480
    cam.set(3, cam_w) # set video width
    cam.set(4, cam_h) # set video height
    
    
    face_detector = cv2.CascadeClassifier(cascade_path)
    # For each person, enter one numeric face id
    #face_id = input('\n enter user id end press <return> ==>  ')
    # face_id = 0
    # print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    #count = 0  #初期化
    
    while(True):
        
        
        
        ret, img_cap = cam.read()
        img = img_cap.copy()
        gray = cv2.cvtColor(img_cap, cv2.COLOR_BGR2GRAY)
        #顔のリストを取得(minSizeは検出最小サイズを指定)
        #参考(http://opencv.jp/opencv-2.1/cpp/object_detection.html)
        
        
        
        #ここから編集部分          2020/12/23
        #ここで四角枠を作成する
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(cam_w // 7, cam_h // 7))
        #print(faces)                   #facesには、枠の4点の数値が記載される　　[[233 121 278 278]]
        
        
        #検出ウィンドウ('image')で、顔を置いてほしい場所に四角形を作る　　
        cv2.rectangle(img, (160, 50), (460, 400), (0, 255, 0), thickness=5)
        
        
        #message = '緑色で表示されている四角枠に顔を収めてください'
        
        #font_path = 'C:/Windows/Fonts/HGRPP1.TTC'
        #font = ImageFont.truetype(font_path, 32)
        
        
    
        
        
        position = (10, 10)
        
        
        
        
        #if len(faces)==1:
        #count =count+1
        im_name = "Image"
        #im_name = "Image_{}_{}".format(str(face_id), str(count))
        #print(im_name)                         #im_name：ファイル名が入る　Image_0_1
        
        #txt_name = im_name + ".txt"
        #im_name = im_name+".jpg"                       #Image_0_1.jpg
        im_name = im_name+".png"
        image = im_name
        im_name = 0
        print(im_name)
        print('ココ')
        #ココ戻すかも　評価用に
        cv2.imwrite(DATA_DIR+image, img_cap)  #face_new + Image_0_1.jpg = face_newImage_0_1.jpg
        
        #return image
            #ここから3行分は、jpgと共に出力されるtxtファイルの中身を表す
            #with open(DATA_DIR + txt_name, mode="w", encoding='utf-8', newline="\n") as f:
            #    for (x, y, w, h) in faces:
            #        f.write("{} {} {} {} {}\n".format(face_id, x/cam_h, y/cam_w, w/cam_w, h/cam_h))
                    
            #return im_name
            
        #else:
        #    faces = ()
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     #基準の線　　ここを基準として下に囲えばプラス、上に囲えばマイナス
            #cv2.rectangle(img, (x,y), (x+w,y-h), (255,0,0), 2)     
        print(image)
        #画像として切り出す実際のウィンドウ　　ウィンドウの名前は、'image'
        cv2.imshow('image', img)
        #print(img)
        
        
        print(len(image))
        
        #ここは削除　　参考までに残す
        #if len(image) != 0:
            #name = detect_who(image)
            
            
        return image
            #return 
        
        k = cv2.waitKey(300) & 0xff
        if k == 27:
            break
        #k = cv2.waitKey(300) & 0xff 
        #if k == 27:     # Press 'ESC' for exiting video
        #    print("Stop")
        #    break
        #elif count >= CAP_COUNT:       #
        #    print("Done!")
        #    break
        # Do a bit of cleanup
    #print("\n [INFO] Exiting Program and cleanup stuff")
    #cam.release()
    #cv2.destroyWindow('image')

        #return image

#この関数は必要ない
#def detect_who(image):
    #print('この関数へ')
    #予測
    #if len(image) != 0:
    #    print('ここに入る')
        
        
        
        
        
        #print(model.predict(image))
        #nameNumLabel=np.argmax(model.predict(image))
        #if nameNumLabel== 0:
        #    name="nakamura"
        #elif nameNumLabel==1:
        #    name="tukamoto"
        #elif nameNumLabel==2:
        #    name="kobayakawa"
        #return name
    #    print('終わり')





def btn_in_click():
    global predict_name
    print('出社押された')
    
    
    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1
    print('OK')
    if value2 == 'tukamoto':
        worksheet.update_cell(13, 3, '出社')
        worksheet.update_cell(13, 4, value3)
    elif value2 == 'nakamura':
        worksheet.update_cell(16, 3, '出社')
        worksheet.update_cell(16, 4, value3)
    elif value2 == 'kobayakawa':                        
        worksheet.update_cell(14, 3, '出社')
        worksheet.update_cell(14, 4, value3)
    
    
    
    
    

    predict_name.set("")
    timer.set("")


def btn_out_click():
    print('退社押された')
    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1

    if value2 == 'tukamoto':
        worksheet.update_cell(13, 3, '')
        worksheet.update_cell(13, 5, value3)
    elif value2 == 'nakamura':
        worksheet.update_cell(16, 3, '')
        worksheet.update_cell(16, 5, value3)
    elif value2 == 'kobayakawa':                        
        worksheet.update_cell(14, 3, '')
        worksheet.update_cell(14, 5, value3)




    predict_name.set("")
    timer.set("")
    
    
def btn_off_click():
    print('休み押された')

    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1
    if value2 == 'tukamoto':
        worksheet.update_cell(13, 3, '休み')
    elif value2 == 'nakamura':
        worksheet.update_cell(16, 3, '休み')
    elif value2 == 'kobayakawa':                        
        worksheet.update_cell(14, 3, '休み')

    

    predict_name.set("")
    timer.set("")


#def btn_change_click():        #名前を変更したい
#    global predict_name, timer, root, p0, p1, p2
#    
#    print('名前変更します')
#    #print(value2)
#    
#    print(predict_name.get())
#    
#    print(p1)
#    time = timer.get()
#    print(time)
#    
#    predict_name.set("")
#    
#    #label_change(time, p0, p1, p2, value2)
#    label_change(time, p0, p1, p2)
#    predict_name.set("")
#    timer.set("")
    
#    root.destroy()
    



def btn_in2_click():
    global change_value
    print('出社押された')
    print(change_value)
    
    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1
    print('OK')
    if change_value == 'tukamoto':
        worksheet.update_cell(13, 3, '出社')
        worksheet.update_cell(13, 4, value3)
    elif change_value == 'nakamura':
        worksheet.update_cell(16, 3, '出社')
        worksheet.update_cell(16, 4, value3)
    elif change_value == 'kobayakawa':                        
        worksheet.update_cell(14, 3, '出社')
        worksheet.update_cell(14, 4, value3)
    
    
    
    
    

    predict_name.set("")
    timer.set("")


def btn_out2_click():
    print('退社押された')
    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1

    if value2 == 'tukamoto':
        worksheet.update_cell(13, 3, '')
        worksheet.update_cell(13, 5, value3)
    elif value2 == 'nakamura':
        worksheet.update_cell(16, 3, '')
        worksheet.update_cell(16, 5, value3)
    elif value2 == 'kobayakawa':                        
        worksheet.update_cell(14, 3, '')
        worksheet.update_cell(14, 5, value3)
        
        
    predict_name.set("")
    timer.set("")
    
        
        
def btn_off2_click():
    print('休み押された')

    worksheet = gc.open_by_key(SPREADSHEET_KEY).sheet1
    if value2 == 'tukamoto':
        worksheet.update_cell(13, 3, '休み')
    elif value2 == 'nakamura':
        worksheet.update_cell(16, 3, '休み')
    elif value2 == 'kobayakawa':                        
        worksheet.update_cell(14, 3, '休み')

    

    predict_name.set("")
    timer.set("")
    




def select(value):
    global change_value
    print(value)       #ここで値が受け渡される
    
    change_value = value
    #predict_name = StringVar(value=value)
    print(change_value)
    
    frame2 = ttk.Frame(root, padding=(0, 0))
    frame2.grid(row=3, column=1)
    
    button1 = ttk.Button(frame2, text='出社', command=btn_in2_click)
    button1.pack(side=LEFT, pady=10)
    
    frame3 = ttk.Frame(root, padding=(0, 0))
    frame3.grid(row=3, column=2)
    
    button2 = ttk.Button(frame3, text='退社', command=btn_out2_click)
    button2.pack(side=LEFT, pady=10)
    
    frame4 = ttk.Frame(root, padding=(0, 0))
    frame4.grid(row=3, column=3)
    
    button3 = ttk.Button(frame4, text='休み', command=btn_off2_click)
    button3.pack(side=LEFT, pady=10)
    
    
    #print(var_option.get())

    
    #predict_name = value
    #print(predict_name)

def message(image, name, time, class0, class1, class2):
    global root, value1, value2, value3, predict_name, timer, c0, c1, c2
    
    
    value1 = image      #画像
    value2 = name       #名前
    value3 = time       #時間
    
    c0 = class0
    c1 = class1
    c2 = class2
    
    class_list = [c0, c1, c2]
    #print(c0)                  #c0~c2まで、各配列要素の中身はきちんと存在する
    #print(class_list[0])
    #print(c1)
    #print(class_list[1])
    #print(c2)
    #print(class_list[2])
    
    
    root = Tk()
    
    root.title('検出結果')

    root.resizable(False, False)
    frame1 = ttk.Frame(root, padding=(64))
    frame1.grid()


    



    
    if value2 == 'nakamura':
        face_image = PhotoImage(file='C:/Users/nakamura-taiki/.conda/envs/yolo_test2/yolo/Image_0_138.png', master=root)
        #face_image = PhotoImage(file=value1, master=root)
    elif value2 == 'tukamoto':
        face_image = PhotoImage(file='C:/Users/nakamura-taiki/.conda/envs/yolo_test2/yolo/Image_1_29.png', master=root)
        #face_image = PhotoImage(file=value1, master=root)
    elif value2 == 'kobayakawa':
        face_image = PhotoImage(file='C:/Users/nakamura-taiki/.conda/envs/yolo_test2/yolo/Image_2_153.png', master=root)
        #face_image = PhotoImage(file=value1, master=root)
        
        
    #画像・名前・時刻　frameに表示
    label1 = ttk.Label(frame1, image=face_image, compound=TOP, padding=(10, 5, 10, 30))
    label1.grid(row=0, column=2)
    
    
    
    
    predict_name = StringVar(value=value2)
    print(predict_name.get())
    #predict_name_entry = ttk.Entry(frame1, textvariable=predict_name, width=20)
    predict_name_optionmenu = tk.OptionMenu(frame1, predict_name, *class_list, command=select)
    #predict_name_optionmenu = tk.OptionMenu(frame1, predict_name, *class_list)
    predict_name_optionmenu.grid(row=1, column=2)
    
    timer = StringVar(value = value3)
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
    

    #frame5 = ttk.Frame(frame1, padding=(0, 0))
    #frame5.grid(row=1, column=3)
    
    #button4 = ttk.Button(frame5, text='名前変更', command=btn_change_click)
    #button4.pack(pady=10)

    
    

    
    root.after(10000, lambda: root.destroy()) #10秒後にリセット
    
    
    root.mainloop()
    
    #if predict_name != "":
    #    change_label = 0
    #    return change_label
    

    
    



    
    

if __name__ == '__main__':
    
    
    image = ""
    
    
    
    while image == "":
        print('start')
        
        
        
        cascade_path = "C:/work/opencv/data/haarcascades/haarcascade_frontalface_alt.xml"
        
        image = image_cut(cascade_path)
        
        cv2.destroyWindow('image')
        
        #name = cnn_predict(model, image)
        
        name, pre0, pre1, pre2, class0, class1, class2 = cnn_evaluate(image)
        
        #nameに関してウィンドウに表示されるようにする  image, name両方使う
        print(name)
        
        date = datetime.datetime.now()
        i = str(math.floor(date.minute/15)*15)
        if i == "0":
            i = "00"
        
        j = str(date.hour) +":" + i
        predict_time = j
        print(predict_time)
        
        a = message(image, name, predict_time, class0, class1, class2)    #画像、名前、時間を渡す
        #detect_who(image)
        #print(a)
        
        #if a == 0:
        #    b = label_change(image, predict_time, pre0, pre1, pre2)
        #else:
        #    print('指定の関数はパスしました')
        #    pass
        
        os.remove("C:/work/face/new/face_new/evaluate/Image.png")
        
        
        
        pre_image = image
        image = ""
        
        pre_name = name
        name = ""
        
        pre_predict_time = predict_time
        predict_time = ""
        
        
    
    
    
    
    
    