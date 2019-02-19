import numpy as np
import scipy.io as sio
import pydot
from keras.layers import Dense, Dropout, Conv2D, Flatten, Input
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

X_train = np.loadtxt('HAPT Data Set/Train/X_train.txt')
Y_train = np.loadtxt('HAPT Data Set/Train/y_train.txt')
ID_train = np.loadtxt('HAPT Data Set/Train/subject_id_train.txt')
X_test = np.loadtxt('HAPT Data Set/Test/X_test.txt')
Y_test = np.loadtxt('HAPT Data Set/Test/y_test.txt')
ID_test = np.loadtxt('HAPT Data Set/Test/subject_id_test.txt')

data = np.expand_dims(np.concatenate((X_train, X_test), axis=0), axis=-1)
labels = np.expand_dims(np.concatenate((Y_train, Y_test), axis=0), axis=-1)
labels = to_categorical(labels, num_classes=int(np.max(labels)+1))
subject_ids = np.expand_dims(np.concatenate((ID_train, ID_test), axis=0), axis=-1)

print('Data shape: ' + str(data.shape))
print('Labels shape: ' + str(labels.shape))
print('Subject IDs shape: ' + str(subject_ids.shape))

measurement_num = len(labels)
time_num = len(data[0, :])
# Standardizes the data
mean_values = np.mean(data, axis=1, keepdims=1)
std_values = np.std(data, axis=1, keepdims=1)
data -= mean_values
data /= std_values

# # Splits the data
X_train, X_test, Y_train, Y_test = train_test_split(data, labels, test_size=0.15, random_state=1)
X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train, test_size=0.15, random_state=1)

# Model Architecture
input = Input((time_num, 1))
x = Dense(20, activation='relu')(input)
x = Flatten()(x)
output = Dense(13, activation='softmax')(x)
model = Model(inputs=[input], outputs=[output])
plot_model(model, to_file='smol_net.png', show_shapes=True)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopper = EarlyStopping(patience=1, verbose=1)
check_pointer = ModelCheckpoint(filepath='smol_net.hdf5', verbose=1, save_best_only=True)
model.fit(X_train, Y_train, batch_size=32, epochs=10, shuffle='true',
          callbacks=[early_stopper, check_pointer], validation_data=(X_dev, Y_dev))
plot_model(model, to_file='smol_net.png', show_shapes=True)

# Loads best loss epoch model
loaded_model = load_model('smol_net.hdf5')
# Evaluates the loaded model
evaluation = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('Evaluation Metrics: ', loaded_model.metrics_names[0], evaluation[0], loaded_model.metrics_names[1], evaluation[1])
# Makes the predictions from the loaded model
predictions = loaded_model.predict(X_test)
