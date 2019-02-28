import numpy as np
import scipy.io as sio
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import itertools
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
    if dataset=='time': # zero padded time data
        data = sio.loadmat('data/data_zeropad.mat')
        data = data['data1']
        data = np.expand_dims(data, axis=-1)
        labels = sio.loadmat('data/y.mat')
        labels = labels['y']
    elif dataset=='time_raw': # raw variable length time data
        data = sio.loadmat('data/data_raw.mat')
        data = data['data']
        labels = sio.loadmat('data/y.mat')
        labels = labels['y']
    elif dataset=='time_slice': # 2 second slices
        data = sio.loadmat('data/data_timeslice.mat')
        data = data['data_timeslice']
        data = np.expand_dims(data, axis=-1)
        labels = sio.loadmat('data/y_timeslice.mat')
        labels = labels['y_timeslice']
    elif dataset=='frequency': # FFT data (1001 samples from 0 to 25 Hz)(only magnitude)
        data = sio.loadmat('data/data_fft.mat')
        data = data['data_fft']
        data = np.expand_dims(data, axis=-1)
        labels = sio.loadmat('data/y.mat')
        labels = labels['y']
    elif dataset=='time_and_frequency': # Zero padded time data and frequency data (magnitude+phase) stacked in a matrix
        data1 = sio.loadmat('data/data_time_and_fft1.mat')
        data1 = data1['data_time_and_fft1']
        data2 = sio.loadmat('data/data_time_and_fft2.mat')
        data2 = data2['data_time_and_fft2']
        data3 = sio.loadmat('data/data_time_and_fft3.mat')
        data3 = data3['data_time_and_fft3']
        data = np.concatenate((data1,data2,data3),axis=0)
        data = np.expand_dims(data, axis=-1)
        labels = sio.loadmat('data/y.mat')
        labels = labels['y']
    elif dataset=='feature': # Extracted features from dataset
        X_train = np.loadtxt('data/HAPT_dataset/Train/X_train.txt')
        Y_train = np.loadtxt('data/HAPT_dataset/Train/y_train.txt')
        X_test = np.loadtxt('data/HAPT_dataset/Test/X_test.txt')
        Y_test = np.loadtxt('data/HAPT_dataset/Test/y_test.txt')
        data = np.expand_dims(np.concatenate((X_train, X_test), axis=0), axis=-1)
        labels = np.expand_dims(np.concatenate((Y_train, Y_test), axis=0), axis=-1)
    labels = to_categorical(labels)
    print('Data shape:', data.shape, 'Labels shape:', labels.shape)
    print('Measurement number:', data.shape[0], 'Time number:', data.shape[1], 'Channel number:', data.shape[2])
    return data, labels


def standardize(data, architecture):
    if architecture == 'lstm':
        data_out = []
        for d in data[0,:]:
            mean_values = np.mean(d, axis=0, keepdims=1)
            d = np.subtract(d, mean_values)
            std_values = np.std(d, axis=0, keepdims=1)
            d = np.divide(d, std_values)
            d = np.expand_dims(d, axis=0)
            data_out.append(d)
        data = data_out
    else:
        mean_values = np.mean(data, axis=1, keepdims=1)
        data = np.subtract(data, mean_values)
        std_values = np.std(data, axis=1, keepdims=1)
        data = np.divide(data, std_values)
    print('Mean value:', np.mean(data))
    print('STD value:', np.mean(np.std(data)))
    return data


def preprocess_data(dataset, architecture):
    # Loads, standardizes, and splits the data into train, dev, test sets
    data, labels = load_data(dataset)
    data = standardize(data, architecture)
    X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.40, random_state=1)
    X_test, X_dev, Y_test, Y_dev = train_test_split(X_test, Y_test, test_size=0.50, random_state=1)
    if architecture != 'lstm':
        print('X_train shape:', X_train.shape)
        print('X_dev shape:', X_dev.shape)
        print('X_test shape:', X_test.shape)
    else:
        Y_train = np.expand_dims(Y_train, axis=1)
        Y_dev = np.expand_dims(Y_dev, axis=1)
        Y_test = np.expand_dims(Y_test, axis=1)
    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test

def filter_num(base_filter_num, c, block_num):
    f_pow = min(c, block_num - c)
    if c > block_num/2:
        f_pow += 1
    return base_filter_num*(2**(f_pow-1))


def percept_leg(input, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop):
    for c1 in range(1, conv1_block+1):
        x = Conv2D(filters=filter_num(base_filter_num, c1, conv1_block), kernel_size=conv1_kernel, activation='relu', padding='same')(input)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,1))(x)
        x = Dropout(drop)(x)
    x = Conv2D(filters=base_filter_num, kernel_size=conv2_kernel, strides=(1,3),  activation = 'relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(drop)(x)
    output = Dense(units=13, activation='softmax')(x)
    return output


def model_architecture(X_train, architecture, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop):
    if architecture=='lstm':
        input = Input((None, 6))
        x = CuDNNLSTM(128, return_sequences=True, input_shape=(None, 6))(input)
        x = CuDNNLSTM(32)(x)
        output = Dense(13, activation='softmax')(x)
    if architecture=='perceptnet':
        input = Input((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        t = Lambda(lambda x: x[:, :, 0:6])(input)
        f = Lambda(lambda x: x[:, :, 6:18])(input)
        t = percept_leg(t, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop)
        f = percept_leg(f, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop)
        x = concatenate([t, f])
        x = Dense(500, activation='relu')(x)
        output = Dense(units=13, activation='softmax')(x)
    elif architecture=='dense':
        input = Input((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        x = Lambda(lambda x: K.squeeze(x, axis=-1))(input)
        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(32, activation='relu')(x)
        output = Dense(13, activation='softmax')(x)
    elif architecture=='small':
        input = Input((X_train.shape[2], X_train.shape[3]))
        x = Dense(20, activation='relu')(input)
        x = Flatten()(x)
        output = Dense(13, activation='softmax')(x)
    model = Model(inputs=[input], outputs=[output])
    plot_model(model, to_file='Network_Figures/'+str(architecture)+'.png', show_shapes=True)
    model.summary()
    return model


def train_model(X_train, Y_train, X_dev, Y_dev, architecture, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop, batch_size, learning_rate):
    model = model_architecture(X_train, architecture, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop)
    opt = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    early_stopper = EarlyStopping(patience=10, verbose=1)
    check_pointer = ModelCheckpoint(filepath='Trained_Networks/network.hdf5', verbose=1, save_best_only=True)
    if architecture != 'lstm':
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=1000, shuffle='true',
              callbacks=[early_stopper, check_pointer], validation_data=(X_dev, Y_dev))
    else:
        model.fit_generator(itertools.cycle(zip(X_train, Y_train)), steps_per_epoch=len(X_train), epochs=1000, shuffle='true',
              callbacks=[early_stopper, check_pointer], validation_data=itertools.cycle(zip(X_dev, Y_dev)), validation_steps=len(X_dev))


def run_experiment(dataset='time', architecture='conv', conv1_block=4, base_filter_num=32, conv1_kernel=(20,1), conv2_kernel=(20,3), drop=0.4, batch_size=32, learning_rate=0.001):
    X_train, X_dev, X_test, Y_train, Y_dev, Y_test = preprocess_data(dataset, architecture)
    train_model(X_train, Y_train, X_dev, Y_dev, architecture, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop, batch_size, learning_rate)
    predictions = evaluate_experiment(X_test, Y_test, architecture)
    return predictions


def evaluate_experiment(X_test, Y_test, architecture):
    loaded_model = load_model('Trained_Networks/network.hdf5')  # Loads best loss epoch model
    if architecture != 'lstm':
        evaluation = loaded_model.evaluate(X_test, Y_test, verbose=0)  # Evaluates the loaded model
        predictions = loaded_model.predict(X_test)  # Makes the predictions from the loaded model
    else:
        evaluation = loaded_model.evaluate_generator(iter(zip(X_test, Y_test)), steps=len(X_test))
        predictions = loaded_model.predict_generator(iter(X_test), steps=len(X_test))
    print('Evaluation Metrics:', loaded_model.metrics_names[0], evaluation[0], loaded_model.metrics_names[1], evaluation[1])  # test loss and accuracy
    os.rename('Trained_Networks/network.hdf5', 'Trained_Networks/'+str(architecture)+'_'+str('%.4f' % evaluation[1])+'.hdf5')
    return predictions


predictions = run_experiment(dataset='time_and_frequency', 
                            architecture='perceptnet', 
                            conv1_block=9,
                            base_filter_num=32, 
                            conv1_kernel=(20,1), conv2_kernel=(20,3), 
                            drop=0.4,
                            batch_size=32,
                            learning_rate=0.001)
