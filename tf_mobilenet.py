#!/usr/bin/python3
# program to implement lightweight mobilenet filter on some images.
# see video at https://www.youtube.com/watch?v=qFJeN9V1ZsI&t=27s
# name: R. Melton
# date: 1/5/21

import os,shutil 
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.layers import Dense, Activation 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import categorical_crossentropy 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.models import Model 
from tensorflow.keras.applications import imagenet_utils 
import itertools 
import os, sys 
import shutil 
import random, time 
import matplotlib.pyplot as pyplot


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
#from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random, time
import matplotlib.pyplot as pyplot
#from tensorflow.keras import backend as kbe

#matplotlib inline

OUTPUT_DATA='/home/user01/g/python3/data/sd/'
#this is where the data currently lives on local disk
TRAIN_PATH=OUTPUT_DATA+'train/'
VALID_PATH=OUTPUT_DATA+'valid/'
TEST_PATH= OUTPUT_DATA+'test/'

#image height and width which mobilenet model expects
IMG_HEIGHT=224
IMG_WIDTH=224

BATCH_SIZE=10

#see if GPU is available
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print('num GPUs ',len(physical_devices))
#if sharing RAM between CPU and GPU, this next step becomes important
if len(physical_devices) > 0:
  tf.config.experimental.set_memory_growth(physical_devices[0],True)


#load mobilenet model
mobile = tf.keras.applications.mobilenet.MobileNet()
mobile.summary()
print('number params: ',mobile.get_weights(),len(mobile.get_weights()))
print('-------------- after params ')

tot = 0
for ar in mobile.get_weights():
  print(' mobile param len ',ar,len(ar))
  tot += len(ar)
print('total params:  ',tot)
# assert params['non_trainable_params' ] == 21888
# assert params['trainable_params'] == 4231976

# prepare image for mobilenet
def prepare_image(file):
  im_path  = 'data/MobileNet-samples/'
  im = image.load_img(im_path+file,target_size=(224,224))
  im_array = image.img_to_array(im)
  im_array_ext_dims = np.expand_dims(im_array,axis=0)
  #scale RGB to values from -1 to 1 in the image
  return tf.keras.applications.mobilenet.preprocess_input(im_array_ext_dims)

def plot_images(image_array):
  fig,axes = plt.subplots(1,10,figsize=(20,20))
  axes = axes.flatten()
  for img, ax in zip(image_array,axes):
    ax.imshow(img)
    ax.axis('off')
  plt.tight_layout()
  plt.show()


def plot_confusion_matrix(cm,classes,
  normalize=False,
  title="confusion matrix",
  cmap=plt.cm.Blues):
  """
  this function prints and plots a confusion matrix
  """
  plt.imshow(cm,interpolation='nearest',cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks,classes,rotation=45)
  plt.yticks(tick_marks.classes)
  
  if normalize:
    cm = cm.astypt('float' / cm.sum(axis=1)[:,np.newaxis])
    print('normalized cm')
  else:
    print('cm without normalization')
  print(cm)


#customize mobilenet model
x = mobile.layers[-6].output 
output = Dense(units=10,activation='softmax')(x)

model = Model(inputs= mobile.input,outputs=output)
#don't retrain the old layers
for layer in model.layers[:-23]:
  layer.trainable = False

model.summary()

model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(target_size(224,224),batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(target_size(224,224),batch_size=10)
test_batches =  ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(target_size(224,224),batch_size=10,shuffle=True)

model.fit(x=train_batches,validation_data=valid_batches,epochs=30,verbose=2)

test_labels = test_batches.classes
predictions = model.predict(x=test_batches,verbose=0)

#confusion matrix, predict digits:
test_labels = test_batches.classes
predictions = model.predict(x=test_batches,verbose=0)

cm = confusion_matrix(y_true=test_labels,y_pred=predictions.argmax(axis=1))
print(test_batches.class_indices)

cm_plot_labels = ['0','1','2','3','4','5','6','7','8','9']

plot_confusion_matrix(cm=cm,classes=cm_plot_labels,title="Confusion Matrix")

print('end of mobilenet program')