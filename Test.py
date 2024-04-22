import cv2
import keras
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
model = keras.models.load_model('my_best_model.epoch10-loss0.80.hdf5')

img=cv2.imread('colordata/fair/fair47.jpg')
img=cv2.resize(img, (32,32))
img = img_to_array(img)
img = np.expand_dims(img, axis = 0)
result = model.predict(img)
result=np.argmax(result[0])

print((result))