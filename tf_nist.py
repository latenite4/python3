#!/usr/bin/python3
# implement TF model which will be used to classify NIST digits using keras.
#
# Name: R. Melton
# date: 1/3/2021
# based on: https://www.youtube.com/watch?v=bee0GrKBCrE
# you will need sudo apt install nvidia-cuda-toolkit

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import time,sys

objects = tf.keras.datasets.mnist    # many images of digits 0-9
(training_images, training_labels),(test_images,test_labels) = objects.load_data()


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

m.compile(optimizer = tf.keras.optimizers.Adam(),
  loss = 'sparse_categorical_crossentropy',
  metrics=['accuracy'])

#train the model
t = time.time()
m.fit(training_images,training_labels,epochs=5) #5 itterations over all data

print(f'training duration: {time.time() - t}s')
start_test_time = time.time()
m.evaluate(test_images,test_labels)
print(f'test duration: {time.time() - start_test_time}s')

#show first image from test data
plt.imshow(test_images[0])
prediction=m.predict(test_images)  # do all test images
print('predicted number: ',np.argmax(prediction[0]))
print(f'\n\ntensorFlow semVer: ',{tf.__version__})
print('python version: ',sys.version_info)


