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
for line in lines:
    source_path = line[0]
    file_path = os.path.join('data', source_path)
    image = cv2.imread(file_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape)
print(y_train.shape)
        

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=6)

model.save('model.h5')
