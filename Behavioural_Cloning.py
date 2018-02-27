
# coding: utf-8

# In[1]:

import csv
import pandas as pd
import numpy as np
import cv2

lines=[]
with open("./data/driving_log.csv") as csvfile:
    next(csvfile)#ignore header
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
images = []
measurements=[]
for line in lines:
    source_path= line[0]
    filename = source_path.split('/')[-1]
    img = cv2.imread("./data/IMG/"+filename)
    images.append(img)
    steering_val = float(line[3])
    measurements.append(steering_val)
y_train = np.array(measurements)
X_train = np.array(images)


# In[2]:

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
batch_size=16
print(len(validation_samples))


# In[12]:

import cv2
import numpy as np
from sklearn.utils import shuffle
import matplotlib.image as mpimg

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                name_left = './data/IMG/'+batch_sample[1].split('/')[-1]
                name_right = './data/IMG/'+batch_sample[2].split('/')[-1]
                
                center_image = mpimg.imread(name)
                left_image = mpimg.imread(name_left)
                right_image = mpimg.imread(name_right)
                
                center_angle = float(batch_sample[3])
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                left_correction = center_angle+0.1
                right_correction = center_angle-0.1
                angles.append(center_angle)
                angles.append(left_correction)
                angles.append(right_correction)
                if(center_angle!=0):
                    image_flipped_c= np.fliplr(center_image)
                    image_flipped_l= np.fliplr(left_image)
                    image_flipped_r= np.fliplr(right_image)
                    images.append(image_flipped_c)
                    images.append(image_flipped_l)
                    images.append(image_flipped_r)
                    angles.append(-1*(center_angle))
                    angles.append(-1*(left_correction))
                    angles.append(-1*(right_correction))                    

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield (X_train, y_train)



# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)


# In[15]:

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Cropping2D,Lambda
from keras.layers.convolutional import Convolution2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0)),input_shape=(160,320,3)))
model.add(Lambda(lambda x:x/255-0.5))
model.add(Convolution2D(24,5,5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
         
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)/batch_size, validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)/batch_size, nb_epoch=5)

    
model.save('model.h5')


# In[ ]:



