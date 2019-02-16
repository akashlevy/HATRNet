import numpy as np
import scipy.io as sio
from keras.layers import Dense, Dropout, Conv2D, Flatten, Input
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot()
# plt.show()

data = sio.loadmat('OurData/data1.mat')
data = data['data1']
labels = sio.loadmat('OurData/y.mat')
labels = labels['y']

print('Data shape:', data.shape)  # (1214, 2032, 6)
print('Labels shape:', labels.shape)  # (1214, 1)

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
data = np.expand_dims(np.divide(np.squeeze(data), std_values), axis=-1)
data = np.squeeze(data)
print('Mean value:', np.mean(np.mean(data, axis=1)))
print('Standard Deviation value:', np.mean(np.std(data, axis=1)))

# Splits the data
train_split = 0.7
dev_split = 0.15
test_split = 0.15
# # Train set
X_train = data[0:int(np.floor(train_split*measurement_num)), :, :]
Y_train = labels[0:int(np.floor(0.7*measurement_num)), :]
# # Evaluation set
X_dev = data[int(np.floor(train_split*measurement_num)):int(np.floor((train_split+dev_split)*measurement_num)), :, :]
Y_dev = labels[int(np.floor(train_split*measurement_num)):int(np.floor((train_split+dev_split)*measurement_num)), :]
# # Test set
X_test = data[int(np.floor((train_split+dev_split)*measurement_num)):, :, :]
Y_test = labels[int(np.floor((train_split+dev_split)*measurement_num)):, :]

print('X_train shape:', X_train.shape)

# Model Architecture
input = Input((time_num, channel_num))
x = Dense(50, activation='relu', input_shape=(time_num, channel_num))(input)
x = Dense(50, activation='relu', input_shape=(time_num, channel_num))(input)
x = Dense(50, activation='relu', input_shape=(time_num, channel_num))(input)

x = Flatten()(x)
output = Dense(13, activation='softmax')(x)
model = Model(inputs=[input], outputs=[output])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopper = EarlyStopping(patience=5, verbose=1)
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
