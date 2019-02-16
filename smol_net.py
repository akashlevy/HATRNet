import numpy as np
import scipy.io as sio
from keras.layers import Dense, Dropout, Conv2D, Flatten, Input
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
# import matplotlib
# matplotlib.use('PS')
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(t, X_train[4, :])
# plt.show()

X_train = np.loadtxt('HAPT Data Set/Train/X_train.txt')
print('DataX_train shape: ' + str(X_train.shape))
Y_train = np.loadtxt('HAPT Data Set/Train/y_train.txt')
ID_train = np.loadtxt('HAPT Data Set/Train/subject_id_train.txt')
X_test = np.loadtxt('HAPT Data Set/Test/X_test.txt')
Y_test = np.loadtxt('HAPT Data Set/Test/y_test.txt')
ID_test = np.loadtxt('HAPT Data Set/Test/subject_id_test.txt')

data = np.expand_dims(np.concatenate((X_train, X_test), axis=0), axis=-1)
labels = np.expand_dims(np.concatenate((Y_train, Y_test), axis=0), axis=-1)
# CHECK UP ON TO CATEGORICAL + 1
labels = to_categorical(labels, num_classes=int(np.max(labels)+1))
subject_ids = np.expand_dims(np.concatenate((ID_train, ID_test), axis=0), axis=-1)

print('Data shape: ' + str(data.shape))
print('Labels shape: ' + str(labels.shape))
print('Subject IDs shape: ' + str(subject_ids.shape))

measurement_num = len(labels)
time_num = len(data[0, :])
mean_values = np.mean(data, axis=1, keepdims=1)
variance_values = np.var(data, axis=1, keepdims=1)

# # Standardizes the data
data -= mean_values
data /= variance_values
# VARIANCE STANDARDIZATION IS NOT WORKING PROPERLY!!!!

# print('Mean Value: ' + str(np.mean(np.mean(data, axis=1))))
# print('Variance Value: ' + str(np.var(data, axis=1, keepdims=1)))

# Splits the data
train_split = 0.7
dev_split = 0.15
test_split = 0.15
# Train set
X_train = data[0:int(np.floor(train_split*measurement_num)), :]
Y_train = labels[0:int(np.floor(0.7*measurement_num))]
# Evaluation set
X_dev = data[int(np.floor(train_split*measurement_num)):int(np.floor((train_split+dev_split)*measurement_num)), :]
Y_dev = labels[int(np.floor(train_split*measurement_num)):int(np.floor((train_split+dev_split)*measurement_num))]
# Test set
X_test = data[int(np.floor((train_split+dev_split)*measurement_num)):, :]
Y_test = labels[int(np.floor((train_split+dev_split)*measurement_num)):]

# Model Architecture
input = Input((time_num, 1))
x = Dense(20, activation='relu', input_shape=(time_num, 1))(input)
x = Flatten()(x)
output = Dense(13, activation='softmax')(x)
model = Model(inputs=[input], outputs=[output])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopper = EarlyStopping(patience=5, verbose=1)
check_pointer = ModelCheckpoint(filepath='net_1.hdf5', verbose=1, save_best_only=True)
model.fit(X_train, Y_train, batch_size=32, epochs=100, shuffle='true',
          callbacks=[early_stopper, check_pointer], validation_data=(X_dev, Y_dev))

# Loads best loss epoch model
loaded_model = load_model('net_1.hdf5')
# Evaluates the loaded model
evaluation = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('Evaluation Metrics: ', loaded_model.metrics_names[0], evaluation[0], loaded_model.metrics_names[1], evaluation[1])
# Makes the predictions from the loaded model
predictions = loaded_model.predict(X_test)
