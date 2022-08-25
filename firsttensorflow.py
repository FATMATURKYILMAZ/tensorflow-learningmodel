# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 08:19:15 2022

@author: ASUS
"""
#veriyi bölüyoruz
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sbn
import matplotlib.pyplot as plt
dataFrame = pd.read_excel("bisiklet_fiyatlari.xlsx")

y=dataFrame["Fiyat"].values
x=dataFrame[["BisikletOzellik1","BisikletOzellik2"]].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=15)

x_train.shape
x_test.shape

#scaling =nöronların boyutunu değiştirmek
scaler=MinMaxScaler()

scaler.fit(x_train)

x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#modeli oluşturduk
model=Sequential()
#burada verilen 4 rakamı modelde ne kadarnöron bulunacağı anlamına geliyor
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))

model.add(Dense(1))

model.compile(optimizer="rmsprop",loss="mse")

#training(öğrenme)
model.fit(x_train,y_train,epochs=250)

loss=model.history.history["loss"]

sbn.lineplot(x=range(len(loss)),y=loss)

#loss değerlerini verir test ve train işlemlerindeki değerlerin yakınlığına bakılır
trainLoss=model.evaluate(x_train,y_train,verbose=0)
testLoss=model.eveluate(x_test,y_test,verbose=0)
trainLoss
testLoss