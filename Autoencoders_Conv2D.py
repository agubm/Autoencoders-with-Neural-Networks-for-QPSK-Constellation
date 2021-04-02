# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 14:29:54 2021
@author: aguboshimec
"""
### General Imports ###
import numpy as np
from numpy import arange
from numpy import argmax, array
import matplotlib.pyplot as plt
import keras
import copy
from keras import backend as K
from tensorflow.keras.models import Model, model_from_json
from keras import layers, regularizers
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, TensorBoard, History, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense,Input, UpSampling1D, Dropout,UpSampling2D, MaxPooling1D, Activation, GaussianNoise,BatchNormalization,  Flatten, ZeroPadding2D, Conv1D, Conv2D, MaxPooling2D, Lambda, Layer
from keras.utils import plot_model, to_categorical
from tensorflow import keras

#parameter and variable declaration:
k = 2          # information bits per symbol eg. QPSK would have 2bitperSymbol.
n = 2          # channel use per message. I'd liken a message to number of symbols of a constellation
M = 2**k         # message could mean/reflect the modulation order or number of symbols.
R = k/n        # effective throughput or communication rate or bits per symbol per channel use

EbNo_dB = -12   # Eb/N0 used for training. this makes trained model robust
noise_stdDev = np.sqrt(1 / (2*R*10**(EbNo_dB/10))) # Noise Standard Deviation
noOfSamples = int(400) #number of samples or observations
epoch = 20
batchSize = 150 
 
data_all = None
test_data = None
test__data_all = None # the copied version
test_data_all = None

#Generate training dataset
def dataset():
    global data_all
    data_all = []
    for i in range (1):
        data = np.random.randint(low=0, high=M, size=(M, ))
        print (data)
        # the convert to one-hot encoded version
       
        data = to_categorical(data, num_classes= M)
        data_all.append(data)        
dataset() 

data_all = array(data_all) #convert to an array or 3D tensor   

#dataset formatting. equally Split dataset into training and validation:
data_all = np.array_split(data_all, 2)

x__train = data_all[1] #training
x__train = x__train.reshape(-1,M,M,1) #reshapes to allowable input dimension
x__validtn = data_all[0] #validation
x__validtn = x__validtn.reshape(-1,M,M,1)


#models the channel for the given noise characteristic
def channel_layer(x, sigma):
    w = K.random_normal(K.shape(x), mean=0.0, stddev=noise_stdDev)
    return x + w

#defines Power Normalization for Tx
def normalization(x):
    mean = K.mean(x ** 2)
    return x / K.sqrt(2 * mean)

# Encoder
input_sym = Input(shape=( M, M, 1))

x = Conv2D(128, (2, 2), padding='same', activation='relu')(input_sym) #236 filters
x = BatchNormalization(name='d_1')(x)
x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
x = Conv2D(128,(2, 2), padding='same', activation='linear')(x)
x = BatchNormalization(name='d_2')(x)
encoded = MaxPooling2D(pool_size=(2,2), padding='same')(x) #more like, this defines 2by2 Channel
 
x = Lambda(normalization, name='power_norm')(x)

x_h_y = Lambda(channel_layer, arguments={'sigma': noise_stdDev}, name='channel_layer')(x)

# Decoder
y = UpSampling2D((2, 2))(x_h_y)
y = Conv2D(128,(2, 2), padding='same', activation='relu')(y)
y = BatchNormalization(name='d_3')(y)
y = Conv2D(1,(2, 2), padding='same', activation='relu')(y)
y = BatchNormalization(name='d_4')(y)
decoded = Activation('sigmoid')(y)

autoencoderConv2D = Model(input_sym, decoded)
autoencoderConv2D.compile(optimizer='Adam', loss = 'binary_crossentropy', metrics=['accuracy'])
autoencoderConv2D.summary()

#display the input and output shapes of each layer:
plot_model(autoencoderConv2D, "autoencoder.png", show_shapes=True)
DeepNNConv2D = autoencoderConv2D.fit(x__train, x__train, validation_data = (x__validtn, x__validtn), epochs=epoch, shuffle = False, batch_size =  batchSize, verbose= 1)

encoder = Model(input_sym, encoded)

def test_dataset():
    global test_data_all, test_data, test__data_all
    test_data_all = []
    test__data_all = []
    for i in range (5000): #generates xx samples
        test_data = np.random.randint(low=0, high=M, size=(M, ))
        test__data = copy.copy(test_data)
        # the convert to one-hot encoded version
        test_data = to_categorical(test_data, num_classes= M)
        test__data_all.append(test__data) #copy of the orignal test data
        test_data_all.append(test_data)        
test_dataset()

# used as test data
test_data_all = array(test_data_all) #convert to an array or 3D tensor   
test_data_all = test_data_all.reshape(-1,M,M,1) #reshapes to fit acceptable dimension of trained model


error_rate_all = []
EbNo_dB_all = []
for EbNo_linear in arange(1,20):
    x_test_noisy = (1/EbNo_linear) + test_data_all #adds noise
    #encoded_data = encoder.predict(xxxxx) #for now, I dont need to show the encoded data or message
    decoded_data = autoencoderConv2D.predict(x_test_noisy)
    
    position = np.argmax(decoded_data, axis=2)
    test__data_all = array(test__data_all)
    x_test_predicted = np.reshape(position, newshape = test__data_all.shape) 
    error_rate = np.mean(np.not_equal(test__data_all,x_test_predicted)) #compares, avergaes, and determines how many errors for each block
    
    error_rate_all.append(error_rate)
    EbNo_dB_all.append(10*np.log10(EbNo_linear)) #converts to dB, and the appends ready for plotting


#plots loss (b oth training and validation) over epoch:
#Error_Rate vs. EbNodB
plt.plot(EbNo_dB_all, error_rate_all)
plt.ylabel('block error rate')
plt.xlabel('EbNo (dB)')
plt.title('Error Rate vs. Eb/No')
plt.grid(b=None, which='major', axis='both')
plt.show()
    
#Loss
plt.plot(DeepNNConv2D.history['loss'])
plt.plot(DeepNNConv2D.history['val_loss'])
plt.title('Graph of final Performance Training Loss')
plt.ylabel('Loss')
plt.xlabel('No. of Epoch')
plt.legend(['Loss', 'Validation Loss'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.show()

#Accuracy
plt.plot(DeepNNConv2D.history['accuracy'])
plt.plot(DeepNNConv2D.history['val_accuracy'])
plt.title('Graph of Performance Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('No. of Epoch')
plt.legend(['Accuracy', 'Validation Accuracy'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.show()


