import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

dataset_train = pd.read_csv("C:\\Users\\kjsj9\\Desktop\\data_0715_20220308.csv")
trainset = dataset_train.iloc[:, 1:2].values

print(trainset)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_scaled = sc.fit_transform(trainset)

x_train = []
y_train = []

for i in range(60,1259):
    x_train.append(training_scaled[i-60:i, 0])
    y_train.append(training_scaled[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

