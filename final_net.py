import numpy as np
import scipy.io as sio
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import itertools
import csv
import random
from random import randint
from keras.layers import Dense, Dropout, Conv2D, Flatten, Input, BatchNormalization, Lambda
from keras.layers import Conv1D, MaxPooling1D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.merge import concatenate
from keras.layers.core import Reshape
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical, plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def load_data(dataset):
    if dataset=='time_and_frequency': # Zero padded time data and frequency data (magnitude+phase) stacked in a matrix
        data = sio.loadmat('data/data_time_and_fft.mat')
        X_train = np.expand_dims(data['X_train'], axis=-1)
        X_dev = np.expand_dims(data['X_dev'], axis=-1)
        X_test = np.expand_dims(data['X_test'], axis=-1)  
        Y_train = to_categorical(data['Y_train'])
        Y_dev = to_categorical(data['Y_dev'])
        Y_test = to_categorical(data['Y_test']) 
    print('X_train shape:', X_train.shape)
    print('X_dev shape:', X_dev.shape)
    print('X_test shape:', X_test.shape)
    print('X_train Mean value:', np.mean(X_train))
    print('X_train STD value:', np.mean(np.std(X_train)))
    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test


def filter_num(base_filter_num, c, block_num):
    f_pow = min(c, block_num - c)
    if c > block_num/2:
        f_pow += 1
    return base_filter_num*(2**(f_pow-1))


def model_architecture(X_train, architecture, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, dense_size, drop):
    if architecture=='perceptnet':
        input = Input((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        t = Lambda(lambda x: x[:, :, 0:6])(input)
        f = Lambda(lambda x: x[:, :, 6:18])(input)
        t = percept_leg(t, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop)
        f = percept_leg(f, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop)
        x = concatenate([t, f])
        x = Dense(dense_size, activation='relu')(x)
        output = Dense(units=13, activation='softmax')(x)
    model = Model(inputs=[input], outputs=[output])
    #plot_model(model, to_file='Network_Figures/'+str(architecture)+'.png', show_shapes=True)
    model.summary()
    return model


def train_model(X_train, Y_train, X_dev, Y_dev, architecture, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, dense_size, drop, batch_size, learning_rate):
    model = model_architecture(X_train, architecture, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, dense_size, drop)
    opt = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    early_stopper = EarlyStopping(patience=5, verbose=1)
    check_pointer = ModelCheckpoint(filepath='Trained_Networks/network.hdf5', verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=100, shuffle='true', callbacks=[early_stopper, check_pointer], validation_data=(X_dev, Y_dev))


def run_experiment(dataset='time_and_frequency', architecture='perceptnet', conv1_block=4, base_filter_num=32, conv1_kernel=(20,1), conv2_kernel=(20,3), dense_size=20, drop=0.4, batch_size=32, learning_rate=0.001, min_loss):
    X_train, X_dev, X_test, Y_train, Y_dev, Y_test = load_data(dataset)
    train_model(X_train, Y_train, X_dev, Y_dev, architecture, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, dense_size, drop, batch_size, learning_rate)
    evaluation, predictions = evaluate_experiment(X_dev, Y_dev, architecture, min_loss) # evaluate on dev set -> hyperparam search
    return evaluation, predictions


def evaluate_experiment(X_test, Y_test, architecture, min_loss):
    loaded_model = load_model('Trained_Networks/network.hdf5')  # Loads best loss epoch model
    evaluation = loaded_model.evaluate(X_test, Y_test, verbose=0)  # Evaluates the loaded model
    if min_loss > evaluation[0]:
        loaded_model.save('Trained_Networks/final_network.hdf5')
    predictions = loaded_model.predict(X_test)  # Makes the predictions from the loaded model
    print('Evaluation Metrics:', loaded_model.metrics_names[0], evaluation[0], loaded_model.metrics_names[1], evaluation[1])  # test loss and accuracy
    #os.rename('Trained_Networks/network.hdf5', 'Trained_Networks/'+str(architecture)+'_'+str('%.4f' % evaluation[1])+'.hdf5')
    os.remove('Trained_Networks/network.hdf5') # in hyperparam search: delete model (hyperparams are saved)
    return evaluation, predictions


conv1_block=2
base_filter_num=60
conv1_kernel=14
conv2_kernel=42
dense_size=71
drop=0.365952189776869
batch_size=16 # (powers of 2)
learning_rate=0.002602647881969
min_loss = 0
max_accuracy = 0
for j in range(20):
    avg_loss = 0
    avg_accuracy = 0

    evaluation, predictions = run_experiment(dataset='time_and_frequency', 
                                             architecture='perceptnet', 
                                             conv1_block=conv1_block,
                                             base_filter_num=base_filter_num, 
                                             conv1_kernel=(conv1_kernel,1), conv2_kernel=(conv2_kernel,3), 
                                             dense_size=dense_size,
                                             drop=drop,
                                             batch_size=batch_size,
                                             learning_rate=learning_rate
                                             min_loss=min_loss)
    print(evaluation)
    if min_loss > evaluation[0]:
        min_loss = evaluation[0]
    if max_accuracy < evaluation[1]:
        max_accuracy = evaluation[1]
    avg_loss += evaluation[0]
    avg_accuracy += evaluation[1]
    K.clear_session() # clears session to prevent slowdown
avg_loss = avg_loss/(j+1)
avg_accuracy = avg_accuracy/(j+1)
print('Average Loss:', avg_loss, 'Average Accuracy:', avg_accuracy)
print('Minimum Loss:', min_loss, 'Max Accuracy:', max_accuracy)
