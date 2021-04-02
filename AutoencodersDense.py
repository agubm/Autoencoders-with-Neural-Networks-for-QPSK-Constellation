# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 22:07:46 2021
Note: Messageblock and Block is used interchangeably. A message contains M number of 'symbols'.
@author: aguboshimec
"""

#General imports
import numpy as np
from numpy import argmax, array, arange
import matplotlib.pyplot as plt
import keras
import copy
from keras import backend as K
from keras import layers, regularizers
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, TensorBoard, History, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense,Input, UpSampling1D, Dropout, MaxPooling1D, Activation, GaussianNoise,BatchNormalization,  Flatten, ZeroPadding2D, Conv1D, Conv2D, MaxPooling2D, Lambda, Layer
from keras.utils import plot_model, to_categorical
from tensorflow import keras

#parameter and variable declaration:
k = 2          # information bits per symbol eg. QPSK would have 2bitperSymbol.
n = 2          # channel use per message. I'd liken a message to number of symbols of a constellation
M = 2**k         # messages could reflect the modulation order or number of symbols, M=4
R = k/n        # effective throughput or communication rate or bits per symbol per channel use

EbNo_dB = -12    # Eb/N0 used for training. #to make model more robust
noise_stdDev = np.sqrt(1 / (2*R*10**(EbNo_dB/10))) # Noise Standard Deviation
noOfSamples = int(40000) #number of sample data generated in the for-loop. It determines the 3 dimension of our matrix or vector.
epoch = 80
batchSize = 150
inputsize = M*M  #since I was considering a dense model. #flattened
data_all = None

#Generate training dataset
def dataset():
    global data_all
    data_all = []
    for i in range (noOfSamples):
        data = np.random.randint(low=0, high=M, size=(M, ))
        # the convert to one-hot encoded version
        data = to_categorical(data, num_classes= M)
        data_all.append(data)  
        
dataset() 

data_all = array(data_all) #convert to an array or 3D tensor   

#data set formatting. equally Splits dataset into training and validation:
data_all = np.array_split(data_all, 2)
x__train = data_all[1] #training
x__validtn = data_all[0] #validation
x_train = np.reshape(data_all[1], (len(x__train),inputsize)) #training
x_validtn = np.reshape(data_all[0], (len(x__validtn),inputsize)) #validation
        

#Building with Keras Sequential, dense model
model = Sequential()
model.add(Dense(8, init = 'random_uniform',activation='relu', input_shape =(inputsize, ), name = "transmitter_input")) #first layer #I used dense layering for now here
model.add(Dense(5 , init = 'uniform', activation='relu'))# Hidden layer

model.add(BatchNormalization()) #ensure ourput of encoder lie between 0 & 1

model.add(GaussianNoise(noise_stdDev, input_shape=(n,)))  #models/mimicks the channel

model.add(Dense(5, init = 'random_uniform', activation='relu'))#Hidden layer, 
model.add(Dense(8, init = 'random_uniform', activation='relu'))#Hidden layer,
model.add(Dense(inputsize, init = 'uniform', activation='sigmoid',  input_shape = (inputsize, ), name = "reciever_output"))  #Output layer,

model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics=['accuracy'])

model.summary() #obtain the summary of the network model

#display the input and output shapes of each layer:
plot_model(model, "autoencoderA.png", show_shapes=True)

#trains model
DeepNN = model.fit(x_train, x_train, validation_data = (x_validtn, x_validtn), epochs=epoch, shuffle = False, batch_size =  batchSize, verbose= 1)


#generate testing dataset or message
x_test_cat_all = []
x__test_all = []
for t in arange(1, 5000):
    x_test = np.random.randint(low=0, high=M, size=(M, )) #testing
    x__test = copy.copy(x_test)
    x_test_cat = to_categorical(x_test, num_classes= M)
    # the convert to one-hot encoded version

    x__test_all.append(x__test) #copy of the orignal test data
    x_test_cat_all.append(x_test_cat)

#predict over a range of snr:
error_rate_all = []
EbNo_dB_all = []

for EbNo_linear in arange(1,20, 1):
     # then convert to one-hot encoded version
    x_test_noisy = (1/EbNo_linear) + x_test_cat_all #add noise
    x_test_reshaped = np.reshape(x_test_noisy,(x_test_noisy.shape[0], M*M)) #x_test_noisy.shape[0] gives the first dim of the tnesor

    decoder_output = model.predict(x_test_reshaped)
    decoder__output = np.reshape(decoder_output,(x_test_noisy.shape[0], M, M))
    # Decode One-Hot vector
    position = np.argmax(decoder__output, axis=2)
    x__test_all = array(x__test_all)
    x_test_predicted = np.reshape(position, newshape = x__test_all.shape)

    error_rate = np.mean(np.not_equal(x__test_all,x_test_predicted)) #compares, and determines how many error for each block
    error_rate_all.append(error_rate)
    EbNo_dB_all.append(10*np.log10(EbNo_linear)) #converts to dB, and the appends ready for plotting
 

plt.plot(EbNo_dB_all, error_rate_all)
plt.ylabel('block error rate')
plt.xlabel('EbNo (dB)')
plt.title('Error Rate vs. Eb/No')
plt.grid(b=None, which='major', axis='both')
plt.show()

#Loss
plt.plot(DeepNN.history['loss'])
plt.plot(DeepNN.history['val_loss'])
plt.title('Graph of final Performance Training Loss')
plt.ylabel('Loss')
plt.xlabel('No. of Epoch')
plt.legend(['Loss', 'Validation Loss'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.show()

#Accuracy
plt.plot(DeepNN.history['accuracy'])
plt.plot(DeepNN.history['val_accuracy'])
plt.title('Graph of Performance Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('No. of Epoch')
plt.legend(['Accuracy', 'Validation Accuracy'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.show()
