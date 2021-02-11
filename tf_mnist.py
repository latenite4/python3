#!/usr/bin/python3
# implement TF model which will be used to classify MNIST digits using keras.
#
# Name: R. Melton
# email: rnm@pobox.com
# date: 1/3/2021
# based on: https://www.youtube.com/watch?v=bee0GrKBCrE
# you will need sudo apt install nvidia-cuda-toolkit

import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import time,sys,os,platform,distro,vers
from tensorflow.python.client import device_lib


#try to install TF 2 if it will run on this HW
# try:
#   tensorflow_version   #<< this function only exists in google colab VM
#   print('installed TF 2.x')
# except Exception:
#   print('could not install TF 2')
#   pass

if __name__ == '__main__':
  vers.show_versions_info()


#see if GPU is available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('num GPUs ',len(physical_devices))
# if len(physical_devices) > 0:
#   tf.config.experimental.set_memory_growth(physical_devices[0],True)

objects = tf.keras.datasets.mnist    # many images of digits 0-9
(training_images, training_labels),(test_images,test_labels) = objects.load_data()

device_lib.list_local_devices()

#print some of the digit images
for i in range(9):
  plt.subplot(330+1+i)
  plt.imshow(training_images[i])

# print dimensions for training data
print(training_images.shape)

#print 1st training image of one digit
print(training_images[0])

#now normalize all training and test input values to 0 - 1
training_images = training_images / 255.0
test_images = test_images / 255.0

m = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(128,activation='relu'), 
  tf.keras.layers.Dense(10,activation=tf.nn.softmax)])

m.summary() #show summary

m.compile(optimizer = tf.keras.optimizers.Adam(),
  loss = 'sparse_categorical_crossentropy',
  metrics=['accuracy'])

#train the model
t = time.time()
m.fit(training_images,training_labels,epochs=5,shuffle=True) #5 itterations over all data


print(f'training duration: {time.time() - t}s')
start_test_time = time.time()
m.evaluate(test_images,test_labels)
print(f'test duration: {time.time() - start_test_time}s')
# write out one hot data values so we know what they are.
# test_images.class_indices

#show first image from test data
plt.imshow(test_images[0])
prediction=m.predict(test_images)  # do all test images
print('predicted number0: ',np.argmax(prediction[0]))
print(': ',prediction[0])
print('predicted number1: ',np.argmax(prediction[1]))
print(': ',prediction[1])
print('predicted number2: ',np.argmax(prediction[2]))
print(': ',prediction[2])
print('predicted number3: ',np.argmax(prediction[3]))
print(': ',prediction[3])
print('predicted number4: ',np.argmax(prediction[4]))
print(': ',prediction[4])



