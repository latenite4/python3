#!/usr/bin/python3
# program to use tensorflow to augment data sets.
# initially tested on signed digits (0-9) in ~/Downloads/data/signed
# name: R. Melton
# date: 1/9/21

import os,shutil ,random,platform,distro
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
from scipy import ndimage,misc
import itertools 
import os, sys 
import shutil 
import random, time 
import matplotlib.pyplot as pyplot
import imageio,platform,vers


def plots(ims,figsize=(12,6),rows=1,interp=False,titles=None):
  if type(ims[0]) is np.ndarray:
    ims= np.array(ims).astype(np.uint8)
    if(ims.shape[-1] != 3):
      ims = ims.transpose((0,2,3,1))
  f = pyplot.figure(figsize=figsize)
  cols = len(ims)//rows if len(ims) %2 == 0 else len(ims)//rows+1
  for i in range(len(ims)):
    sp = f.add_subplot(rows,cols,i+1)
    sp.axis('Off')
    if titles is not None:
      sp.set_title(titles[i],fontsize=16)
    pyplot.imshow(ims[i],interpolation=None if interp else 'none')



#show picture on console of images
def plot_images(images_array):
  fig,axes = pyplot.subplots(1,10,figsize=(20,20))
  axes = axes.flatten()
  for img ,ax in zip(images_array,axes):
      ax.imshow(img)
      ax.axis('off')
  pyplot.tight_layout()
  pyplot.show()


INPUT_DATA='/home/user01/Downloads/data/sd/Sign-Language-Digits-Dataset/Dataset'
OUTPUT_DATA='/home/user01/g/python3/data/sd/'
TRAIN_DATA='/home/user01/g/python3/data/sd/train'

TEMP_DIR='/home/user01/g/python3/temp'
#this is where the data currently lives on local disk
TRAIN_PATH=OUTPUT_DATA+'/train/'
#number to images to add to each digit directory (0-9)
NUM_AUGMENT=10
NUM_DIGITS=10

if __name__ == '__main__':

  #or use: python -c "from tensorflow import keras; print(keras.__version__)"
  vers.show_versions_info()

  gen  = ImageDataGenerator(rotation_range=10,width_shift_range=0.1,
      height_shift_range=0.1,shear_range=0.15, zoom_range=0.1,
      channel_shift_range=10., horizontal_flip=True)

  image_path = '/home/user01/g/python3/data/sd/train/0/IMG_5991.JPG'
  image = np.expand_dims(imageio.imread(image_path),0)

  pyplot.imshow(image[0])
  #print('image name: ',image)
  aug_iter = gen.flow(image)
  aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
  plots(aug_images,figsize=(20,7),rows=2)




  
