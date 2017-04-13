import csv
import os

import cv2
import numpy as np

samples = []
with open('data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            correction = 0.2*np.array([0, +1, -1])
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = line[i].strip()
                    file_path = os.path.join('data', source_path)
                    image = cv2.imread(file_path)
                    measurement = float(line[3]) + correction[i]
                    images.append(image)
                    measurements.append(measurement)
                    images.append(cv2.flip(image, 1))
                    measurements.append(measurement*-1)

            X_train = np.array(images)
            y_train = np.array(measurements)
            print(X_train.shape)
            print(y_train.shape)
            yield shuffle(X_train, y_train)
       
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

#LeNet
#model.add(Convolution2D(6, 5, 5, activation='relu'))
#model.add(MaxPooling2D())
#model.add(Convolution2D(6, 5, 5, activation='relu'))
#model.add(MaxPooling2D())
#model.add(Flatten())
#model.add(Dense(128))
#model.add(Dense(84))
#model.add(Dense(1))

#Nvidia
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, subsample=(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=len(train_samples),
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples), 
                                     nb_epoch=6)

### print the keys contained in the history object
print(history_object.history.keys())

import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
