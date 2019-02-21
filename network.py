import numpy as np
import scipy.io as sio
import tensorflow as tf
import pydot
import os
from keras.layers import Dense, Dropout, Conv2D, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling2D
from keras.layers.merge import concatenate
from keras.layers.core import Reshape
from keras.models import Model, load_model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical, plot_model
from sklearn.model_selection import train_test_split


def load_data(dataset):
    if dataset=='end_to_end':
        data = sio.loadmat('Data/data1.mat')
        data = data['data1']
        data = np.expand_dims(data, axis=-1)
        labels = sio.loadmat('Data/y.mat')
        labels = labels['y']
    elif dataset=='feature':
        X_train = np.loadtxt('Data/HAPT Data Set/Train/X_train.txt')
        Y_train = np.loadtxt('Data/HAPT Data Set/Train/y_train.txt')
        X_test = np.loadtxt('Data/HAPT Data Set/Test/X_test.txt')
        Y_test = np.loadtxt('Data/HAPT Data Set/Test/y_test.txt')
        data = np.expand_dims(np.concatenate((X_train, X_test), axis=0), axis=-1)
        labels = np.expand_dims(np.concatenate((Y_train, Y_test), axis=0), axis=-1)
    labels = to_categorical(labels)
    print('Data shape:', data.shape, 'Labels shape:', labels.shape)
    print('Measurement number:', data.shape[0], 'Time number:', data.shape[1], 'Channel number:', data.shape[2])
    return data, labels


def standardize(data):
    mean_values = np.mean(data, axis=1, keepdims=1)
    data = np.subtract(data, mean_values)
    std_values = np.std(data, axis=1, keepdims=1)
    data = np.divide(data, std_values)
    print('Mean value:', np.mean(data))
    print('STD value:', np.mean(np.std(data)))
    return data


def preprocess_data(dataset):
    # Loads, standardizes, and splits the data into train, dev, test sets
    data, labels = load_data(dataset)
    data = standardize(data)
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.15, random_state=1)
    X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.40, random_state=1)
    print('X_train shape:', X_train.shape)
    print('X_dev shape:', X_dev.shape)
    print('X_test shape:', X_test.shape)
    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test


def model_architecture(X_train, architecture):
    if architecture=='conv':
        input = Input((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        x = Conv2D(16, (12, 3), activation='relu', padding='same')(input)
        x = Dropout(0.4)(x)
        x = Conv2D(32, (12, 3), activation='relu', padding='same')(x)
        x = Dropout(0.4)(x)
        x = Conv2D(32, (12, 3), activation='relu', padding='same')(x)
        x = Dropout(0.4)(x)
        x = Conv2D(16, (12, 3), activation='relu', padding='same')(x)
        x = Flatten()(x)
        output = Dense(13, activation='softmax')(x)
    elif architecture=='dense':
        input = Input((X_train.shape[1], X_train.shape[2]))
        x = Dense(32, activation='relu')(input)
        x = Dense(32, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Flatten()(x)
        output = Dense(13, activation='softmax')(x)
    elif architecture=='small':
        input = Input((X_train.shape[1], X_train.shape[2]))
        x = Dense(20, activation='relu')(input)
        x = Flatten()(x)
        output = Dense(13, activation='softmax')(x)
    model = Model(inputs=[input], outputs=[output])
    plot_model(model, to_file='Network_Figures/'+str(architecture)+'.png', show_shapes=True)
    model.summary()
    return model


def train_model(X_train, Y_train, X_dev, Y_dev, architecture):
    model = model_architecture(X_train, architecture)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    early_stopper = EarlyStopping(patience=1, verbose=1)
    check_pointer = ModelCheckpoint(filepath='Trained_Networks/network.hdf5', verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, batch_size=32, epochs=1, shuffle='true',
              callbacks=[early_stopper, check_pointer], validation_data=(X_dev, Y_dev))


def evaluate_experiment(X_test, Y_test, architecture):
    loaded_model = load_model('Trained_Networks/network.hdf5')  # Loads best loss epoch model
    evaluation = loaded_model.evaluate(X_test, Y_test, verbose=0)  # Evaluates the loaded model
    print('Evaluation Metrics:', loaded_model.metrics_names[0], evaluation[0], loaded_model.metrics_names[1], evaluation[1])  # test loss and accuracy
    os.rename('Trained_Networks/network.hdf5', 'Trained_Networks/'+str(architecture)+'_'+str('%.4f' % evaluation[1])+'.hdf5')
    predictions = loaded_model.predict(X_test)  # Makes the predictions from the loaded model
    return predictions


def run_experiment(dataset='end_to_end', architecture='conv'):
    X_train, X_dev, X_test, Y_train, Y_dev, Y_test = preprocess_data(dataset)
    train_model(X_train, Y_train, X_dev, Y_dev, architecture)
    predictions = evaluate_experiment(X_test, Y_test, architecture)
    return predictions


######################################
# Dataset = feature                  #
#     architecture = small, dense    #
# Dataset = end_to_end               #
#     architecture = conv            #
######################################

predictions = run_experiment(dataset='feature', architecture='small')
