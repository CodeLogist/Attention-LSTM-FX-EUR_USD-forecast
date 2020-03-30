# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time 
from sklearn.preprocessing import MinMaxScaler
from keras import layers
import keras
import keras.backend as K
from sklearn.metrics import mean_squared_error
from math import sqrt

from utils import *
from attention import *

FILE_NAME='data/EURUSD_H1_2010-2019.csv'

data = np.array(pd.read_csv(FILE_NAME,encoding='utf-16',sep='\t')['Close'].values[:], dtype="float32")
lstm_size = 128
dense_size = 256
time_step = 30
batch_size = 64
output_size = 1
input_size = 1
shift = 1
if __name__ == '__main__':

    mode = 'test' #train , test

    print('Forex Attention-LSTM Neural net for EURUSD')

    dataX,dataY = make_data_windowed(data,time_step,shift)
    
    train_split_idx=data.shape[0]*80//100
    
    tdata,labels,testdata,testlabels = train_test_split(dataX, dataY,train_split_idx)

    attention = Attention(lstm_size, batch_size, time_step, input_size,output_size)

    if mode == 'train':
        train=True
        attention_model=attention.model(train)
        attention_model.summary()
        vdata=testdata[-250:]
        vlabels=testlabels[-250:]
        EPOCHS=5
        hist=attention_model.fit(tdata,labels,validation_data=(vdata,vlabels),batch_size=batch_size,verbose=1,epochs=EPOCHS)
        attention_model.save('model/modelv2.h5')

    elif mode == 'test':
        train=False
        attention_model=attention.model(train)
        preds=attention_model.predict(testdata)

        print('------------------------------------------------------------------')
        p_close=[]
        columns = [
            "Pred-Close",
            "Orig-Close",
        ]
        print("{}\t\t{}".format(*columns))
        for i in range(len(preds)):
            p_close.append(preds[i])
            output = [
                preds[i][0],
                testlabels[i][0]
            ]
            print("{:0.5f}\t\t{:0.5f}".format(*output))
        print('----------------------------------------- RMSE: -------------------------------------------')
        rmse = sqrt(mean_squared_error(testlabels,preds))
        print("RMSE: ",rmse)
        
        print('----------------------------------------- Plot( Predictions vs Real values): -------------------------------------------')
        plt.figure(figsize=(15,5))
        plt.plot(preds,label='Prediction')
        plt.plot(testlabels,label='Real')
        plt.legend()
        plt.show()



