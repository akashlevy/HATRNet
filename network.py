import numpy as np
import scipy.io as sio
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import itertools
from keras.layers import Dense, Dropout, Conv2D, Flatten, Input, BatchNormalization, Lambda, TimeDistributed
from keras.layers import Conv1D, MaxPooling1D, MaxPooling2D, GlobalAveragePooling2D, CuDNNLSTM, LSTM
from keras.layers.merge import concatenate
from keras.layers.core import Reshape
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical, plot_model
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} ) 
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)

def load_data(dataset):
    if dataset=='time':
        data = sio.loadmat('data/data_zeropad.mat')
        data = data['data1']
        data = np.expand_dims(data, axis=-1)
        labels = sio.loadmat('data/y.mat')
        labels = labels['y']
    elif dataset=='time_raw':
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
    elif dataset=='frequency':
        data = sio.loadmat('data/data_fft.mat')
        data = data['data_fft']
        data = np.expand_dims(data, axis=-1)
        labels = sio.loadmat('data/y.mat')
        labels = labels['y']
    elif dataset=='time_and_frequency':
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
    elif dataset=='feature':
        X_train = np.loadtxt('data/HAPT_dataset/Train/X_train.txt')
        Y_train = np.loadtxt('data/HAPT_dataset/Train/y_train.txt')
        X_test = np.loadtxt('data/HAPT_dataset/Test/X_test.txt')
        Y_test = np.loadtxt('data/HAPT_dataset/Test/y_test.txt')
        data = np.expand_dims(np.concatenate((X_train, X_test), axis=0), axis=-1)
        labels = np.expand_dims(np.concatenate((Y_train, Y_test), axis=0), axis=-1)
    labels = to_categorical(labels)
    print('Data shape:', data.shape, 'Labels shape:', labels.shape)
    if dataset != 'time_raw':
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


def model_architecture(X_train, architecture):
    if architecture=='conv':
        input = Input((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        x = Conv2D(16, (12, 3), activation='relu', padding='same')(input)
        x = MaxPooling2D(pool_size=(2,1))(x)
        x = Dropout(0.4)(x)
        x = Conv2D(32, (12, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2,1))(x)
        x = Dropout(0.4)(x)
        x = Conv2D(64, (12, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2,1))(x)
        x = Dropout(0.4)(x)
        x = Conv2D(32, (12, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2,1))(x)
        x = Dropout(0.4)(x)
        x = Conv2D(16, (12, 3), activation='relu', padding='same')(x)
        x = Flatten()(x)
        x = Dense(500, activation='relu')(x)
        output = Dense(13, activation='softmax')(x)
    elif architecture=='lstm':
        input = Input((None, 6))
        x = CuDNNLSTM(128, return_sequences=True, input_shape=(None, 6))(input)
        x = CuDNNLSTM(32)(x)
        output = Dense(13, activation='softmax')(x)
    elif architecture=='late_fusion':
        input = Input((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        x = Lambda(lambda x: K.squeeze(x, axis=-1))(input)
        x = Conv1D(filters=32, kernel_size=20, activation='relu', padding='same', data_format='channels_first')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.4)(x)
        x = Conv1D(filters=128, kernel_size=20, activation='relu', padding='same', data_format='channels_first')(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(0.4)(x)
        x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
        x = Conv2D(filters=32, kernel_size=(15,3), strides=(1,3),  activation = 'relu', padding='same')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x)
        # x = Flatten()(x)
        output = Dense(units=13, activation='softmax')(x)
    elif architecture=='perceptnet':
        input = Input((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        x = Conv2D(filters=32, kernel_size=(20,1), activation='relu', padding='same')(input)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,1))(x)
        x = Dropout(0.4)(x)
        x = Conv2D(filters=64, kernel_size=(20,1), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,1))(x)
        x = Dropout(0.4)(x)
        x = Conv2D(filters=128, kernel_size=(20,1), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,1))(x)
        x = Dropout(0.4)(x)
        x = Conv2D(filters=256, kernel_size=(20,1), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,1))(x)
        x = Dropout(0.4)(x)
        x = Conv2D(filters=128, kernel_size=(20,1), activation = 'relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,1))(x)
        x = Dropout(0.4)(x)
        x = Conv2D(filters=64, kernel_size=(20,3), strides=(1,3),  activation = 'relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x)
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


def train_model(X_train, Y_train, X_dev, Y_dev, architecture):
    model = model_architecture(X_train, architecture)
    opt = Adam(lr=0.00075) if architecture=='lstm' else 'adam'
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    early_stopper = EarlyStopping(patience=10, verbose=1)
    check_pointer = ModelCheckpoint(filepath='Trained_Networks/network.hdf5', verbose=1, save_best_only=True)
    if architecture != 'lstm':
        model.fit(X_train, Y_train, batch_size=32, epochs=1000, shuffle='true',
              callbacks=[early_stopper, check_pointer], validation_data=(X_dev, Y_dev))
    else:
        model.fit_generator(itertools.cycle(zip(X_train, Y_train)), steps_per_epoch=len(X_train), epochs=1000, shuffle='true',
              callbacks=[early_stopper, check_pointer], validation_data=itertools.cycle(zip(X_dev, Y_dev)), validation_steps=len(X_dev))


def evaluate_experiment(X_test, Y_test, architecture):
    loaded_model = load_model('Trained_Networks/network.hdf5')  # Loads best loss epoch model
    if architecture != 'lstm':
        evaluation = loaded_model.evaluate(X_test, Y_test, verbose=0)  # Evaluates the loaded model
    else:
        evaluation = loaded_model.evaluate_generator(iter(zip(X_test, Y_test)), steps=len(X_test))
    print('Evaluation Metrics:', loaded_model.metrics_names[0], evaluation[0], loaded_model.metrics_names[1], evaluation[1])  # test loss and accuracy
    os.rename('Trained_Networks/network.hdf5', 'Trained_Networks/'+str(architecture)+'_'+str('%.4f' % evaluation[1])+'.hdf5')
    if architecture != 'lstm':
        predictions = loaded_model.predict(X_test)  # Makes the predictions from the loaded model
    else:
        predictions = loaded_model.predict_generator(iter(X_test), steps=len(X_test))
    return predictions

def plot_confusion_matrix(Y_true, Y_pred, architecture):
    if architecture == 'lstm':
        Y_true = np.squeeze(Y_true, axis=1)
    matrix = confusion_matrix(Y_true.argmax(axis=1), Y_pred.argmax(axis=1))
    plt.figure()
    plt.imshow(matrix, interpolation='nearest')
    plt.colorbar()
    fmt = 'd'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt), horizontalalignment="center", color="white")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    plt.savefig('Results/confmat_%s.png' % architecture)

def run_experiment(dataset='time', architecture='conv'):
    X_train, X_dev, X_test, Y_train, Y_dev, Y_test = preprocess_data(dataset, architecture)
    train_model(X_train, Y_train, X_dev, Y_dev, architecture)
    predictions = evaluate_experiment(X_test, Y_test, architecture)
    plot_confusion_matrix(Y_test, predictions, architecture)
    return predictions



######################################
# Dataset = feature                  #
#     architecture = small, dense    #
# Dataset = time                     #
#     architecture = conv            #
# Dataset = frequency                #
#     architecture = conv            #
######################################

predictions = run_experiment(dataset='time_raw', architecture='lstm')
