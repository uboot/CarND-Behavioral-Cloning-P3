import csv
import os

import cv2
import numpy as np

lines = []
with open('data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    next(reader)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
correction = 0.2*np.array([0, +1, -1])
for line in lines[:2]:
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
       

from keras.models import Sequential
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
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6)

model.save('model.h5')
