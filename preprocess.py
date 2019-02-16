import numpy as np
import scipy.io as sio
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = sio.loadmat('data/data1.mat')
data = data['data1']
labels = sio.loadmat('data/y.mat')
labels = labels['y']
# experiment = sio.loadmat('data/exp.mat')
# experiment = experiment['exp']
# user = sio.loadmat('data/user.mat')
# user = user['user']

print('Data shape:', data.shape)  # (1214, 2032, 6)
print('Labels shape:', labels.shape)  # (1214, 1)
# print('Experiment shape:', experiment.shape)  # (1214, 1)
# print('User shape:', user.shape)  # (1214, 1)

# data = np.expand_dims(data, axis=-1)
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
print('Mean value:', np.mean(np.mean(data, axis=1)))
print('Standard Deviation value:', np.mean(np.std(data, axis=1, keepdims=1)))

# Splits the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=1)
