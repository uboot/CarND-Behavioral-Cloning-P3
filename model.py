import csv
import os

import cv2
import numpy as np

### read the training data and store the augmented data in a list
samples = []
with open('data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    correction = 0.2*np.array([0, +1, -1])
    for line in reader:        
        for view in range(3):
            for flipped in [1, -1]:
                source_path = os.path.basename(line[view].strip())
                file_path = os.path.join('data', 'IMG', source_path)
                
                # this is a hack for comma based floating point numbers which
                # were output by the simulator on my system (German session on
                # openSUSE Linux)
                if len(line[4]) == 1:
                    angle = float(line[3]) # 0 -> 0
                else:
                    angle = float(line[3] + '.' + line[4]) # 0,123 -> 0.123
                    
                data = {
                    'path': file_path,
                    'angle': flipped * (angle + correction[view]),
                    'flipped': flipped
                }
                samples.append(data)

### sub-sample the data
samples = samples[::5]

### split the sub-sampled data in training and validation data
from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

### this generator functions reads and pre-processes the actual training images
### for a given (small) subset of the input samples 
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for sample in batch_samples:
                file_path = sample['path']
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                flipped = sample['flipped']
                measurement = sample['angle']
                
                if flipped == -1:
                    images.append(cv2.flip(image, 1))
                else:
                    images.append(image)
                measurements.append(measurement)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)
            
### create generators which output subsets of the training and the validation
### data, respectively
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)


### create the CNN
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()

### crop and normalize the images...
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))

### ... and setup the Nvidia architecture
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

### train the model
model.compile(loss='mse', optimizer='adam')
print (model.summary())
history_object = model.fit_generator(train_generator,
                                     samples_per_epoch=len(train_samples),
                                     validation_data=validation_generator,
                                     nb_val_samples=len(validation_samples), 
                                     nb_epoch=3)

### print the history and save the model
print(history_object.history.keys())
model.save('model.h5')


### finally, plot the training and validation loss for each epoch
import matplotlib.pyplot as plt

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

