

# import cv2

# import numpy as np
# import random
# from imutils import paths
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# import json
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.utils import to_categorical
# from sklearn import metrics
# import pickle
# import pandas as pd
# import numpy as np
# import os
# from glob import glob
# import random
# import matplotlib.pylab as plt
# from tensorflow.keras.models import model_from_json

# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# import keras
# from keras.utils.np_utils import to_categorical
# from keras.models import Sequential
# from keras.preprocessing.image import ImageDataGenerator

# from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
# from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D
# from tensorflow.keras.callbacks import EarlyStopping


# data = []
# label = []
# print("[INFO] loading images...")
# img_dir=sorted(list(paths.list_images("colordata")))
# random.shuffle(img_dir)
# print("[INFO]  Preprocessing...")
# count=0
# for i in img_dir:
#     img = cv2.imread(i)
#     img=cv2.resize(img, (200,200))
#     imgdata=img_to_array(img)
#     data.append(imgdata)
#     lb=i.split(os.path.sep)[-2]
#     if(lb=="dark"):
#         label.append(0)
#     elif(lb=="avg"):
#         label.append(1)
#     else:
#         label.append(2)

# traindata=np.array(data)
# trainlabel=np.array(label)
# print(traindata)
# X_train, X_test, y_train, y_test = train_test_split(traindata, trainlabel, test_size=0.25, random_state=42)
# print(X_train.shape)
# print(y_train.shape)




# ##Creating a CNN Model
# model = Sequential()
# inputShape = (200, 200,3)
# ##First Convolution Layer
# model.add(Conv2D(32, (5, 5), padding="same",input_shape=inputShape))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(rate=0.25))
# ##Second Convolution Layer
# model.add(Conv2D(32, (5, 5), padding="same"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(rate=0.25))
# ##Third Convolution Layer
# model.add(Conv2D(64, (5, 5), padding="same"))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# model.add(Dropout(rate=0.25))
# ##flattening the output
# model.add(Flatten())
# ##adding Denser layer of 500 nodes
# model.add(Dense(500))
# model.add(Activation("relu"))
#  ##softmax classifier
# model.add(Dense(3))
# model.add(Activation("softmax"))
# model.summary()

# model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# # model = Sequential()

# # # Convolutional layer and maxpool layer 1
# # model.add(Conv2D(32,(3,3),activation='relu',input_shape=(64,64,3)))
# # model.add(MaxPool2D(2,2))

# # # Convolutional layer and maxpool layer 2
# # model.add(Conv2D(64,(3,3),activation='relu'))
# # model.add(MaxPool2D(2,2))

# # # Convolutional layer and maxpool layer 3
# # model.add(Conv2D(128,(3,3),activation='relu'))
# # model.add(MaxPool2D(2,2))

# # # Convolutional layer and maxpool layer 4
# # model.add(Conv2D(128,(3,3),activation='relu'))
# # model.add(MaxPool2D(2,2))

# # # This layer flattens the resulting image array to 1D array
# # model.add(Flatten())

# # # Hidden layer with 512 neurons and Rectified Linear Unit activation function 
# # model.add(Dense(512,activation='relu'))

# # # Output layer with single neuron which gives 0 for Cat or 1 for Dog 
# # #Here we use sigmoid activation function which makes our model output to lie between 0 and 1
# # model.add(Dense(2,activation='sigmoid'))


# # model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])



# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_test, y_test),
#     epochs= 10,
#     batch_size=10
#     )
# # Y_pred = model.predict(X_test)
# # print(Y_pred)
# # # serialize model to JSON
# # model_json = model.to_json()
# # with open("CNNmodel.json", "w") as json_file:
# #        json_file.write(model_json)
# # # serialize weights to HDF5
# # model.save_weights("CNNmodelw.h5")
# # print("Saved model to disk")


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
print("[INFO] splitting datas for training...")
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)
# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=3)
testY = to_categorical(testY, num_classes=3)
print("[INFO]  Training Started...")
print(len(trainY))
print(len(trainX))
print(np.array(trainY).shape)
print(np.array(trainX).shape)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# training the model for 10 epochs
model.fit(np.array(trainX), np.array(trainY), batch_size=32, epochs=10, validation_data=(np.array(testX), np.array(testY)),callbacks=callbacks)
# # serialize model to JSON
# model_json = model.to_json()
# with open("ch_model.json", "w") as json_file:
#         json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("ch_model.h5")
# print("[INFO] Saved model to disk")


