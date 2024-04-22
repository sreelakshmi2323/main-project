from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import  Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from imutils import paths
import numpy as np
import random
import cv2
import os
from keras.callbacks import ModelCheckpoint
filepath = 'my_best_model.epoch{epoch:02d}-loss{val_accuracy:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor="val_accuracy",
                             verbose=1, 
                             save_best_only=True,
                             mode="max")
callbacks = [checkpoint]
##Creating a CNN Model
model = Sequential()
inputShape = (32, 32,3)
##First Convolution Layer
model.add(Conv2D(32, (5, 5), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(rate=0.25))
##Second Convolution Layer
model.add(Conv2D(32, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(rate=0.25))
##Third Convolution Layer
model.add(Conv2D(64, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(rate=0.25))
##flattening the output
model.add(Flatten())
##adding Denser layer of 500 nodes
model.add(Dense(500))
model.add(Activation("relu"))
 ##softmax classifier
model.add(Dense(3))
model.add(Activation("softmax"))
model.summary()
print(model.summary())
data = []
labels = []
print("[INFO] loading images...")
img_dir=sorted(list(paths.list_images("colordata")))
random.shuffle(img_dir)
for i in img_dir:
        img = cv2.imread(i)
        img=cv2.resize(img, (32,32))
        img = img_to_array(img)
        data.append(img)
        lb=i.split(os.path.sep)[-2]

        if(lb=="dark"):
            labels.append(0)
        elif(lb=="avg"):
            labels.append(1)
        else:
            labels.append(2)
        
print(len(data))
print(len(labels))




