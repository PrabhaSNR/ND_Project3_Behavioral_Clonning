import os
import csv
import math

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from scipy import ndimage

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = list()
            angles = list()
            for batch_sample in batch_samples:
                center_path = batch_sample[0].split('/')[-1]#'.data/IMG/'+
                left_path = batch_sample[1].split('/')[-1]
                right_path = batch_sample[2].split('/')[-1]
                #Reading Image data
                center_image = ndimage.imread(center_path,flatten=False,mode='RGB')
                left_image = ndimage.imread(left_path,flatten=False,mode='RGB')
                right_image = ndimage.imread(right_path,flatten=False,mode='RGB')
                #Reading steering wheel angle
                center_angle = float(batch_sample[3])
                left_angle=center_angle+0.20
                right_angle=center_angle-0.20
                #Appending to list
                images.extend([center_image,left_image,right_image])
                angles.extend([center_angle,left_angle,right_angle])
                #Augmenting the data and appending to list
                aug_center_image = cv2.flip(center_image,1)
                aug_left_image = cv2.flip(left_image,1)
                aug_right_image = cv2.flip(right_image,1)
                images.extend([aug_center_image,aug_left_image,aug_right_image])
                angles.extend([-1.0*center_angle, -1.0*left_angle, -1.0*right_angle])
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format

from keras.models import Sequential, Model 
from keras.layers import Lambda , Cropping2D, Dense , Flatten , Convolution2D
import matplotlib.pyplot as plt

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Cropping2D(cropping=((70,25), (0,0)),input_shape=(row, col, ch)))
model.add(Lambda(lambda x: x/127.5 - 1.))
#model.add(Lambda(lambda x: x/127.5 - 1.,
 #       input_shape=(row, col, ch),
 #       output_shape=(row, col, ch)))
model.add(Convolution2D(24,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(36,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(48,(5,5),strides=(2,2),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
#model.add(... finish defining the rest of your model architecture here ...)

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples)/batch_size),validation_data=validation_generator,   validation_steps=math.ceil(len(validation_samples)/batch_size), epochs=5, verbose=1)


### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')