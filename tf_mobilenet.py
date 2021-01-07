#!/usr/bin/python3
# program to implement lightweight mobilenet filter on some images.
# see video at https://www.youtube.com/watch?v=qFJeN9V1ZsI&t=27s
# name: R. Melton
# date: 1/5/21


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activatoin
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion _matrix
import itertools
import os
import shutil
import random, time
import matplotlib.pyplot as pyplot

matplotlib inline


#see if GPU is available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('num GPUs ',len(physical_devices))
#if sharing RAM between CPU and GPU, this next step becomes important
if len(physical_devices) > 0:
  tf.config.experimental.set_memory_growth(physical_devices[0],True)


#load mobilenet model
mm = tf.keras.applications.mobilenet.MobileNet()
mm.summary()
mm_params = count_params(mm)
assert mm_params['non_trainable_params' ] == 21888
assert mm_params['trainable_params'] == 4231976

# prepare image for mobilenet
def prepare_image(file):
  im_path  = 'data/MobileNet-samples/'
  im = image.load_img(im_path+file,target_size=(224,224))
  im_array = image.img_to_array(im)
  im_array_ext_dims = np.expand_dims(im_array,axis=0)
  #scale RGB to values from -1 to 1 in the image
  return tf.keras.applications.mobilenet.preprocess_input(im_array_ext_dims)

from IPython.display import Image
Image(filename='data/MobileNet-samples/1.PNG', width=300,height= 200)

processed_im = prepare_image('1.PNG')

predictions = mm.predict(processed_im)
results = imagenet_utils.decode_predictions(predictions)
results

assert results[0][0][1] == 'American_chameleon'

Image(filename='data/MobileNet-samples/2.PNG', width=300,height= 200)
processed_im = prepare_image('2.PNG')
predictions = mm.predict(processed_im)
results = imagenet_utils.decode_predictions(predictions)
results

assert results[0][0][1] == 'espresso'

Image(filename='data/MobileNet-samples/3.PNG', width=300,height= 200)
processed_im = prepare_image('3.PNG')
predictions = mm.predict(processed_im)
results = imagenet_utils.decode_predictions(predictions)
results

assert results[0][0][1] == 'strawberry'



