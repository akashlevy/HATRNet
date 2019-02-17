import numpy as np
import scipy.io as sio
from keras.layers import Dense, Dropout, Conv2D, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling2D
from keras.layers.merge import concatenate
from keras.layers.core import Reshape
from keras.models import Model, load_model, Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot()
# plt.show()

data = sio.loadmat('OurData/data1.mat')
data = data['data1']
labels = sio.loadmat('OurData/y.mat')
labels = labels['y']

print('Data shape:', data.shape)
print('Labels shape:', labels.shape)

labels = to_categorical(labels)
measurement_num = len(labels)
time_num = len(data[0, :, 0])
channel_num = len(data[0, 0, :])

print('Data shape:', data.shape)
print('Labels shape:', labels.shape)
print('Measurement number:', measurement_num)
print('Time number:', time_num)
print('Channel number:', channel_num)

# Standardizes the data
mean_values = np.mean(data, axis=1, keepdims=1)
print('Mean values shape:', mean_values.shape)
data = np.subtract(data, mean_values)
std_values = np.std(data, axis=1, keepdims=1)
data = np.divide(np.squeeze(data), std_values)
# ADDED FOR CONV
data = np.expand_dims(data, axis=-1)
print('Mean value:', np.mean(np.mean(data, axis=1)))
print('Standard Deviation value:', np.mean(np.std(data, axis=1)))

# Splits the data
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.15, random_state=1)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.15, random_state=1)
print('X_train shape:', X_train.shape)
print('X_dev shape:', X_dev.shape)
print('X_test shape:', X_test.shape)

# Model Architecture
input = Input((time_num, channel_num, 1))
x = Conv2D(16, (12, 6), activation='relu', padding='same', input_shape=(time_num, channel_num, 1))(input)
x = Dropout(0.4)(x)
x = Conv2D(32, (12, 3), activation='relu', padding='same')(x)
x = Dropout(0.4)(x)
x = Conv2D(32, (12, 3), activation='relu', padding='same')(x)
x = Dropout(0.4)(x)
x = Conv2D(16, (12, 6), activation='relu', padding='same')(x)


# input = Input((time_num, channel_num))
# x = Dense(32, activation='relu', input_shape=(time_num, channel_num))(input)
# x = Dense(32, activation='relu')(x)
# x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
# x = Dense(64, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# x = Dense(32, activation='relu')(x)
x = Flatten()(x)
output = Dense(13, activation='softmax')(x)
model = Model(inputs=[input], outputs=[output])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopper = EarlyStopping(patience=3, verbose=1)
check_pointer = ModelCheckpoint(filepath='net_six_channels.hdf5', verbose=1, save_best_only=True)
model.fit(X_train, Y_train, batch_size=32, epochs=100, shuffle='true',
          callbacks=[early_stopper, check_pointer], validation_data=(X_dev, Y_dev))

# Loads best loss epoch model
loaded_model = load_model('net_six_channels.hdf5')
# Evaluates the loaded model
evaluation = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('Evaluation Metrics:', loaded_model.metrics_names[0], evaluation[0], loaded_model.metrics_names[1], evaluation[1])
# Makes the predictions from the loaded model
predictions = loaded_model.predict(X_test)




# model = Sequential()
# model.add(Conv1D(filters=150, kernel_size=15, activation='relu', input_shape=(time_num, channel_num)))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.4))
# model.add(Conv1D(filters=150, kernel_size=15, activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.4))
# layer2_output_shape = model.output_shape
# model.add(Reshape((layer2_output_shape[1], layer2_output_shape[2], -1)))
# model.add(Conv2D(filters=150, kernel_size=(3, 15), strides=(3, 1), activation='relu'))
# model.add(GlobalAveragePooling2D())
# model.add(Dropout(0.4))
# model.add(Dense(units=13, activation='softmax'))


