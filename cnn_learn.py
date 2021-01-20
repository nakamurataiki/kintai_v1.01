# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:57:28 2021

@author: nakamura-taiki
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:49:01 2021

@author: nakamura-taiki
"""


import os
from tensorflow.keras.layers import Input, Dense
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten,MaxPooling2D,Conv2D
from keras import optimizers

import matplotlib.pyplot as plt

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

def cnn_learning(image):
    classes = ['tukamoto', 'kobayakawa','nakamura']
    nb_classes = len(classes)
    batch_size_for_data_generator = 20
    
    base_dir = "C:/Users/nakamura-taiki/.conda/envs/yolo_test2/yolo/"
    
    train_dir = os.path.join(base_dir, 'train')
    #validation_dir = os.path.join(base_dir, 'validation_images')
    test_dir = os.path.join(base_dir, 'test')
    
    train_nakamura_dir = os.path.join(train_dir, 'nakamura')
    train_tukamoto_dir = os.path.join(train_dir, 'tukamoto')
    train_kobayakawa_dir = os.path.join(train_dir, 'kobayakawa')
    
    
    #validation_daisy_dir = os.path.join(validation_dir, 'nakamura')
    #validation_dandelion_dir = os.path.join(validation_dir, 'tukamoto')
    #validation_rose_dir = os.path.join(validation_dir, 'kobayakawa')
    
    
    test_nakamura_dir = os.path.join(test_dir, 'nakamura')
    test_tukamoto_dir = os.path.join(test_dir, 'tukamoto')
    test_kobayakawa_dir = os.path.join(test_dir, 'kobayakawa')
    
    
    # 画像サイズ
    img_rows, img_cols = 128, 96
    
    
    print('total training nakamura images:', len(os.listdir(train_nakamura_dir)),train_nakamura_dir)
    print('total training tukamoto images:', len(os.listdir(train_tukamoto_dir)),train_tukamoto_dir)
    print('total training kobayakawa images:', len(os.listdir(train_kobayakawa_dir)),train_kobayakawa_dir)
    
    
    
    print('total test nakamura images:', len(os.listdir(test_nakamura_dir)),test_nakamura_dir)
    print('total test tukamoto images:', len(os.listdir(test_tukamoto_dir)),test_tukamoto_dir)
    print('total test kobayakawa images:', len(os.listdir(test_kobayakawa_dir)),test_kobayakawa_dir)
    
    
    train_datagen = ImageDataGenerator(rescale=1.0 / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
    
    train_generator = train_datagen.flow_from_directory(directory=train_dir,target_size=(img_rows, img_cols),color_mode='rgb',classes=classes,class_mode='categorical',batch_size=batch_size_for_data_generator,shuffle=True)
    
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        
    validation_generator = test_datagen.flow_from_directory(directory=test_dir,target_size=(img_rows, img_cols),color_mode='rgb',classes=classes,class_mode='categorical',batch_size=batch_size_for_data_generator,shuffle=True)
    
    #今まで使用していた学習アルゴリズム
    #model=Sequential()
    #model.add(Conv2D(32,(3,3),activation='relu',input_shape=(img_rows, img_cols, 3)))
    #model.add(MaxPooling2D((2,2)))
    #model.add(Dropout(0.5))
    #model.add(Conv2D(64,(3,3),activation='relu'))
    #model.add(MaxPooling2D((2,2)))
    #model.add(Dropout(0.5))
    #model.add(Conv2D(128,(3,3),activation='relu'))
    #model.add(MaxPooling2D((2,2)))
    #model.add(Dropout(0.5))
    #model.add(Conv2D(128,(3,3),activation='relu'))
    #model.add(MaxPooling2D((2,2)))
    #model.add(Conv2D(128,(3,3),activation='relu'))
    #model.add(MaxPooling2D((2,2)))
    #model.add(Flatten())
    #model.add(Dropout(0.5))
    #model.add(Dense(512,activation='relu'))
    #model.add(Dense(nb_classes,activation='softmax'))
    
    input_shape=(128,96,3)
    
    model = Sequential()
    model.add(Conv2D(input_shape=input_shape,filters=32,kernel_size=(3, 3),strides=(1, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                     strides=(1, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                     strides=(1, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("sigmoid"))
    model.add(Dense(128))
    model.add(Activation('sigmoid'))
    # 分類したい人数を入れる
    model.add(Dense(len(classes)))
    model.add(Activation('softmax'))
    
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy',optimizer='sgd', metrics=['acc'])
    
    history = model.fit(train_generator,steps_per_epoch=25,epochs=50,validation_data=validation_generator,validation_steps=10,verbose=1) 
    
    
    model.save("C:/Users/nakamura-taiki/.conda/envs/yolo_test2/yolo/cnn/my_model_second.h5")
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()
    
    test_dir = os.path.join(base_dir, 'evaluate')
    test_generator = test_datagen.flow_from_directory(directory=test_dir,target_size=(img_rows, img_cols),color_mode='rgb',classes=classes,class_mode='categorical',batch_size=batch_size_for_data_generator)
    
    test_loss, test_acc = model.evaluate(test_generator, steps=68)
    print('test loss:', test_loss)
    print('test acc:', test_acc)
    
    return model
    
def cnn_evaluate(image):
    
    model = load_model("C:/Users/nakamura-taiki/.conda/envs/yolo_test2/yolo/cnn/my_model_second.h5")
    
    evaluate_dir = "C:/work/face/new/face_new/evaluate/"     #切り取った画像がある場所
    
    classes = ['tukamoto', 'kobayakawa','nakamura']
    
    filename = os.path.join(evaluate_dir, image)            #切り取った画像がある場所とその画像名を足して一つのパスにする＝filenameという変数
    print(filename)
    
    img = np.array(Image.open(filename))                   #そのパスから画像を開いて表示する
    plt.imshow(img)
    
    
    
    
    img = load_img(filename, target_size=(128, 96))       #その画像を指定のサイズでロード
    x = img_to_array(img)                                 #その画像についての特長量を配列にする
    x = np.expand_dims(x, axis=0)                         #特長量配列の次元を追加する
    
    
    predict = model.predict(preprocess_input(x))         #画像の特長量に関する配列をpredictとする
    print(predict)
    for pre in predict:                                  #predict[]の中の変数を順番に見る
        #print(pre[0])
    
        pre_array_0 = pre[0]        #精度　　数値[0]
        pre_array_1 = pre[1]        #          [1]
        pre_array_2 = pre[2]        #          [2]
        
        classes0 = classes[0]       #ラベル名[0]
        classes1 = classes[1]       #      [1]
        classes2 = classes[2]       #      [2]
        
    
    
        y = pre.argmax()                                 #1番大きい数字の場所をyとし、検出した名前とする
        print("test result=",classes[y], pre)
            
        return classes[y], pre_array_0, pre_array_1, pre_array_2, classes0, classes1, classes2                                #その名前を戻り値とする
