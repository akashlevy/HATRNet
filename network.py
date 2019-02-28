import numpy as np
import scipy.io as sio
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import itertools
<<<<<<< HEAD
from keras.layers import Dense, Dropout, Conv2D, Flatten, Input, BatchNormalization, Lambda, TimeDistributed
from keras.layers import Conv1D, MaxPooling1D, MaxPooling2D, GlobalAveragePooling2D, CuDNNLSTM, LSTM
=======
from keras.layers import Dense, Dropout, Conv2D, Flatten, Input, BatchNormalization, Lambda
from keras.layers import Conv1D, MaxPooling1D, MaxPooling2D, GlobalAveragePooling2D, CuDNNLSTM
>>>>>>> parent of c8c9773... commit
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

#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} )
#sess = tf.Session(config=config)
#keras.backend.set_session(sess)


def load_data(dataset):
    if dataset=='time': # zero-padded time data
        data = sio.loadmat('data/data1.mat')
        data = data['data1']
        data = np.expand_dims(data, axis=-1)
        labels = sio.loadmat('data/y.mat')
        labels = labels['y']
    elif dataset=='time_slice': # 2 second slices (no zero-padding required)
        data = sio.loadmat('data/data_timeslice.mat')
        data = data['data_timeslice']
        data = np.expand_dims(data, axis=-1)
        labels = sio.loadmat('data/y_timeslice.mat')
        labels = labels['y_timeslice']
    elif dataset=='frequency': # frequency and phase data
        data = sio.loadmat('data/data_fft.mat')
        data = data['data_fft']
        data = np.expand_dims(data, axis=-1)
        labels = sio.loadmat('data/y.mat')
        labels = labels['y']
    elif dataset=='time_and_frequency': # time, frequency and phase data
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
    elif dataset=='feature': # UCI feature selected data
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


def plot_confusion_matrix(Y_true, Y_pred):
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
    output = Dropout(drop)(x)
    return output


def model_architecture(X_train, architecture, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop, dense_num):
    if architecture=='saimese_perceptnet':
        input = Input((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        t = Lambda(lambda x: x[:, :, 0:5])(input)
        f = Lambda(lambda x: x[:, :, 6:17])(input)
        t = percept_leg(t, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop)
        f = percept_leg(f, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop)
        x = concatenate([t, f])
        x = Dense(dense_num, activation='relu')(x)
        output = Dense(units=13, activation='softmax')(x)
    elif architecture=='perceptnet':
        input = Input((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        x = percept_leg(input, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop)
        x = Dense(dense_num, activation='relu')(x)
        output = Dense(units=13, activation='softmax')(x)
    elif architecture=='lstm':
        input = Input((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        x = Lambda(lambda x: K.squeeze(x, axis=-1))(input)
        x = CuDNNLSTM(128, return_sequences=True, input_shape=(None, 6))(x)
        x = CuDNNLSTM(32)(x)
        output = Dense(13, activation='softmax')(x)
    elif architecture=='conv':
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
    elif architecture=='lstm2':
        input = Input((X_train.shape[1], X_train.shape[2], X_train.shape[3]))
        x = Lambda(lambda x: K.squeeze(x, axis=-1))(input)
        x = CuDNNLSTM(128, return_sequences=True, input_shape=(None, 6))(x)
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


def train_model(X_train, Y_train, X_dev, Y_dev, architecture, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop, dense_num):
    model = model_architecture(X_train, architecture, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop, dense_num)
    opt = Adam(lr=0.0005) if architecture=='lstm' else 'adam'
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    early_stopper = EarlyStopping(patience=5, verbose=1)
    check_pointer = ModelCheckpoint(filepath='Trained_Networks/network.hdf5', verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, batch_size=32, epochs=1, shuffle='true',
              callbacks=[early_stopper, check_pointer], validation_data=(X_dev, Y_dev))
<<<<<<< HEAD
    else:
        model.fit_generator(itertools.cycle(zip(X_train, Y_train)), steps_per_epoch=len(X_train), epochs=1000, shuffle='true',
              callbacks=[early_stopper, check_pointer], validation_data=itertools.cycle(zip(X_dev, Y_dev)), validation_steps=len(X_dev))
=======
>>>>>>> parent of c8c9773... commit


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

def run_experiment(dataset='time_and_frequency', architecture='saimese_perceptnet', conv1_block=4, base_filter_num=32, conv1_kernel=(20,1), conv2_kernel=(20,3), drop=0.4, dense_num=128):
    X_train, X_dev, X_test, Y_train, Y_dev, Y_test = preprocess_data(dataset)
    train_model(X_train, Y_train, X_dev, Y_dev, architecture, conv1_block, base_filter_num, conv1_kernel, conv2_kernel, drop, dense_num)
    predictions = evaluate_experiment(X_test, Y_test, architecture)
    plot_confusion_matrix(Y_test, predictions, architecture)
    return predictions


####################################################################
## Dataset = feature
##     architecture = small, dense
## Dataset = time, time_slice, frequency
##     architecture = perceptnet, conv, late_fusion
## Dataset = time_and_frequency
##     architecture = saimese_perceptnet
####################################################################


predictions = run_experiment(dataset='time_and_frequency',
                            architecture='saimese_perceptnet',
                            conv1_block=4,
                            base_filter_num=32,
                            conv1_kernel=(20,1), conv2_kernel=(20,3),
                            drop=0.4,
                            dense_num=128)
