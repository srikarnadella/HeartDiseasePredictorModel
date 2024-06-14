# -*- coding: utf-8 -*-
"""
Created on Fri Feb 4 11:08:33 2022

@author: 832748
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import OneHotEncoder
from keras import models
from keras import layers
import sklearn.model_selection as model_selection
from sklearn.preprocessing import StandardScaler
from keras import optimizers
#data reading

def createData():
    fileName = 'heart.csv'
    raw_data = open(fileName, 'rt')
    dataOne = np.loadtxt(raw_data, usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12), skiprows = 1, delimiter=",", dtype=str)
    x = dataOne[:,:-1]
    y = dataOne[:,-1]
    y = np.reshape(y, (-1, 1))
    x,y = dataClean(x,y)
    return(x,y)

def dataClean(x,y):
    #YES & NOS
    x = (x[:,:])
    y = (y[:,:])
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    return (x,y)

def skLearning(x,y):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y,train_size=0.75,test_size=0.25, random_state=101)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    return(x_train, y_train, x_test, y_test)

def modelmaker(x_train,y_train,x_test,y_test):
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape = (1,12) ))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizers.RMSprop(lr=0.005), loss='mse', metrics=['mae','accuracy'])
    history = model.fit(x_train, y_train, epochs= 5, batch_size=5, validation_data=(x_test,y_test))
    mae_history = history.history['mae']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    return(model,history,mae_history,acc,val_acc,loss,val_loss)


def testData(x_test,y_test,model):
    test_mae_score = model.evaluate(x_test, y_test)
    return(test_mae_score)



def plotTrainingData(mae_history,acc,val_acc,loss,val_loss):
    epochs = range(1, len(acc) + 1)
    fig, ax = plt.subplots()
    ax.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    ax.plot(epochs, val_loss, 'b', label='Validation loss')
    ax.set(xlabel='Epochs', ylabel='Loss',
           title='Training and validation loss');
    ax.legend()
    acc_values = acc
    val_acc_values = val_acc
    
    fig1, ax1 = plt.subplots()
    ax1.plot(epochs, acc, 'ro', label='Training acc')
    ax1.plot(epochs, val_acc, 'b', label='Validation acc')
    ax1.set(xlabel='Epochs', ylabel='Accuracy', title='Training and validation accuracy');
    ax1.legend()
    
    plt.show()
    return



sc = StandardScaler()
x,y = createData()
trainX , trainY , testX , testY = skLearning(x,y)
model,history , mae_history , acc , val_acc , loss , val_loss = modelmaker(trainX,trainY,testX,testY)
test_mae_score = testData(testX, testY, model)
plotTrainingData(mae_history, acc,val_acc, loss, val_loss)
print ("test_mae_score : " + str(test_mae_score))


