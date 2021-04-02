# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 22:07:46 2021
Note: 
Message and Block is used interchangeably. A message contains M number of 'symbols'.
Mdoel training with same batch-size, epoch, number of (test and train) samples.
Sigmoid >> Softmax; M = 4PSK
@author: aguboshimec
"""

#General lib. imports
import keras
import copy
import numpy as np
from numpy import argmax, array, arange
import matplotlib.pyplot as plt
from keras import backend as K
from keras import layers, regularizers
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, TensorBoard, History, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense,Input, UpSampling1D, Dropout, UpSampling2D, MaxPooling1D, Activation, GaussianNoise,BatchNormalization,  Flatten, ZeroPadding2D, Conv1D, Conv2D, MaxPooling2D, Lambda, Layer
from keras.utils import plot_model, to_categorical
from tensorflow import keras


#parameter and variable declaration:
k = 2          # information bits per symbol eg. QPSK would have 2bitperSymbol.
n = 2          # channel use per message. I'd liken a message to number of symbols of a constellation
M = 2**k         # messages could reflect the modulation order or number of symbols.
R = k/n        # effective throughput or communication rate or bits per symbol per channel use

EbNo_dB = -12    # Eb/N0 used for training. #to make model more robust
noise_stdDev = np.sqrt(1 / (2*R*10**(EbNo_dB/10))) # Noise Standard Deviation
noOfSamples = int(4000) #number of sample data generated in the for-loop. It determines the 3 dimension of our matrix or vector.
epoch = 80
batchSize = 100
inputsize = M*M  #since I was considering a dense model. I kinda flattened
data_all = None
test__data_all = None # the copied version
test_data_all = None

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

#main for Dense Autoencoder
x_train = np.reshape(data_all[1], (len(x__train),inputsize)) #training
x_validtn = np.reshape(data_all[0], (len(x__validtn),inputsize)) #validation

#main for conv2D Autoencoder
x__train = x__train.reshape(-1,M,M,1) #reshapes to allowable input dimension
x__validtn = x__validtn.reshape(-1,M,M,1)


############## generate testing dataset #################
x_test_cat_all = []
x__test_all = []
test_data_all = []
test__data_all = []

for t in range(1, 5000):
    x_test = np.random.randint(low=0, high=M, size=(M, ))
    test_data = copy.copy(x_test) #conv2D. simply to use identical testing data sample
    test__data = copy.copy(test_data)
    # the convert to one-hot encoded version
    test_data = to_categorical(test_data, num_classes= M)
    test__data_all.append(test__data) #copy of the orignal test data
    test_data_all.append(test_data) 
    
    x__test = copy.copy(x_test) #Dense
    # the convert to one-hot encoded version
    x_test_cat = to_categorical(x_test, num_classes= M)
    x__test_all.append(x__test) #copy of the orignal test data
    x_test_cat_all.append(x_test_cat)
    

DeepNN = None
error_rate_allD = None
EbNo_dB_allD = None

def autoencoderDense():   
    global DeepNN, error_rate_allD, EbNo_dB_allD, x__test_all
    #Building with Keras Sequential, dense model
    model = Sequential()
    model.add(Dense(8, init = 'random_uniform',activation='relu', input_shape =(inputsize, ), name = "transmitter_input")) #first layer #I used dense layering for now here
    model.add(Dense(5 , init = 'uniform', activation='relu'))# Hidden layer
    
    model.add(BatchNormalization()) #ensure ourput of encoder lie between 0 & 1
    
    model.add(GaussianNoise(noise_stdDev, input_shape=(n,)))  #models the channel
    
    model.add(Dense(5, init = 'random_uniform', activation='relu'))#Hidden layer, 
    model.add(Dense(8, init = 'random_uniform', activation='relu'))#Hidden layer,
    model.add(Dense(inputsize, init = 'uniform', activation='sigmoid',  input_shape = (inputsize, ), name = "reciever_output"))  #Output layer,
    
    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    
    model.summary() #obtain the summary of the network model
    
    #display the input and output shapes of each layer:
    plot_model(model, "autoencoderA.png", show_shapes=True)
    
    #trains model
    DeepNN = model.fit(x_train, x_train, validation_data = (x_validtn, x_validtn), epochs=epoch, shuffle = False, batch_size =  batchSize, verbose= 1)
    
    #predict over a range of snr:
    error_rate_allD = []
    EbNo_dB_allD = []
    
    for EbNo_linear in arange(1,20):
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
        error_rate_allD.append(error_rate)
        EbNo_dB_allD.append(10*np.log10(EbNo_linear)) #converts to dB, and the appends ready for plotting
     
autoencoderDense()

error_rate_allConv2D =None
EbNo_dB_allConv2D = None
DeepNNConv2D = None

##AutoEncoder with Convoultion Layer, conv2D
def autoencoderConv2D():
    global error_rate_allConv2D, EbNo_dB_allConv2D, DeepNNConv2D, test_data_all, test__data_all
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
    
    encoder = Model(input_sym, encoded) #the encoded version not show yet
    
    # used as test data
    test_data_all = array(test_data_all) #convert to an array or 3D tensor   
    test_data_all = test_data_all.reshape(-1,M,M,1) #reshapes to fit acceptable dimension of trained model
    
    
    error_rate_allConv2D = []
    EbNo_dB_allConv2D = []
    for EbNo_linear in range(1,20):
        x_test_noisy = (1/EbNo_linear) + test_data_all #adds noise
        #encoded_data = encoder.predict(xxxxx) #for now, I dont need to show the encoded data or message
        decoded_data = autoencoderConv2D.predict(x_test_noisy)
        
        position = np.argmax(decoded_data, axis=2)
        test__data_all = array(test__data_all)
        x_test_predicted = np.reshape(position, newshape = test__data_all.shape) 
        error_rate = np.mean(np.not_equal(test__data_all,x_test_predicted)) #compares, avergaes, and determines how many errors for each block
        
        error_rate_allConv2D.append(error_rate)
        EbNo_dB_allConv2D.append(10*np.log10(EbNo_linear)) #converts to dB, and the appends ready for plotting

autoencoderConv2D()

#plots loss (b oth training and validation) over epoch:
    
#Error_Rate vs. EbNodB
plt.plot(EbNo_dB_allD, error_rate_allD)
plt.plot(EbNo_dB_allConv2D, error_rate_allConv2D)
plt.ylabel('block error rate')
plt.xlabel('EbNo (dB)')
plt.title('Error Rate vs. Eb/No')
plt.legend(['conv2D', 'Dense'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.show()
  
#Loss
plt.plot(DeepNNConv2D.history['loss'])
plt.plot(DeepNNConv2D.history['val_loss'])
plt.plot(DeepNN.history['loss'])
plt.plot(DeepNN.history['val_loss'])
plt.title('Graph of final Performance Training Loss')
plt.ylabel('Loss')
plt.xlabel('No. of Epoch')
plt.legend(['conv2D Loss', 'conv2D Validation Loss', 'Dense Loss', 'Dense Validation Loss'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.show()

#Accuracy
plt.plot(DeepNNConv2D.history['accuracy'])
plt.plot(DeepNNConv2D.history['val_accuracy'])
plt.plot(DeepNN.history['accuracy'])
plt.plot(DeepNN.history['val_accuracy'])
plt.title('Graph of Performance Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('No. of Epoch')
plt.legend(['conv2D Accuracy', 'conv2D Validation Accuracy', 'Dense Accuracy', 'Dense Validation Accuracy'], loc='upper right')
plt.grid(b=None, which='major', axis='both')
plt.show()


