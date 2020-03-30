import numpy as np

def make_data_windowed(data,time_step,shift):
    dataX=[]
    dataY=[]
    for i in range(0, len(data) - time_step, shift):
        a = data[i:(i+time_step)]
        dataX.append(a)
        dataY.append(data[i+time_step])
    dataX=np.array(dataX)
    dataY=np.array(dataY)
    return dataX, dataY

def train_test_split(dataX,dataY,train_split_idx):

    tdata=dataX[:train_split_idx]
    labels=dataY[:train_split_idx]
    testdata=dataX[train_split_idx:]
    testlabels=dataY[train_split_idx:]

    labels=np.reshape(labels,(labels.shape[0],1))
    tdata=np.expand_dims(tdata,axis=2)
    testlabels=np.reshape(testlabels,(testlabels.shape[0],1))
    testdata=np.expand_dims(testdata,axis=2)
    
    return tdata,labels,testdata,testlabels 