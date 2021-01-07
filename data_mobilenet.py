#!/usr/bin/python3
#this python3 program will organize the signed digit samples for
# mobilenet model training
#2 testing
#3 validation
# name: R. Melton
# date: 1/6/21
# refer to deeplizard video https://www.youtube.com/watch?v=qFJeN9V1ZsI&t=7873s at 2:15:58
#copy unpreprocessed images with git clone https://github.com/ardamavi/Sign-Language-Digits-Dataset.git

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

#matplotlib inline


#constants for this program
INPUT_DATA='/home/user01/Downloads/data/signed_digits/Sign-Language-Digits-Dataset/Dataset/'
OUTPUT_DATA='/home/user01/g/python3/data/sd/'
EXEC_DIR='/home/user01/g/python3'
#this is where the data currently lives on local disk
TRAIN_PATH=OUTPUT_DATA+'/train/'
VALID_PATH=OUTPUT_DATA+'/valid/'
TEST_PATH=OUTPUT_DATA+'/test/'
#image height and width which mobilenet model expects
IMG_HEIGHT=224
IMG_WIDTH=224

BATCH_SIZE=10

# prepare image for mobilenet
def prepare_image(file):
  im_path  = 'data/MobileNet-samples/'
  im = image.load_img(im_path+file,target_size=(224,224))
  im_array = image.img_to_array(im)
  im_array_ext_dims = np.expand_dims(im_array,axis=0)
  #scale RGB to values from -1 to 1 in the image
  return tf.keras.applications


if __name__ == '__main__':

  img=''
  os.chdir(OUTPUT_DATA)
  #have we run this script before?
  try:
    if os.path.isdir(OUTPUT_DATA+'/0') is False:
      print('reformatting data for mobilenet')

      os.chdir(EXEC_DIR)
      os.mkdir('data/sd/train')
      os.mkdir('data/sd/valid')
      os.mkdir('data/sd/test')

      for i in range(0,10):
        print(f'...reformatting data digit: {i}')
        # make sample input for validity testing
        print(f'source dir: {INPUT_DATA}{i}/')
        valid_samples = random.sample(os.listdir(f'{INPUT_DATA}{i}/'),30)
        print(f'valid samples for: {i} {valid_samples}')
        #move valid images
        for v in valid_samples:
          shutil.move(f'{INPUT_DATA}{i}/{v}',f'{OUTPUT_DATA}/valid/{i}/{v}')

    else:
      print('data formatting has already been done; you don\'t need to do it again')

    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input, \
      directory=TRAIN_PATH,target_size=(IMG_HEIGHT,IMG_WIDTH),batch_size=BATCH_SIZE)
    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
      directory=VALID_PATH,target_size=(IMG_HEIGHT,IMG_WIDTH),batch_size=BATCH_SIZE)
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
      directory=TEST_PATH,target_size=(IMG_HEIGHT,IMG_WIDTH),batch_size=BATCH_SIZE,shuffle=False)

    #make sure counts are correct...
    assert train_batches.n == 1712
    assert valid_batches.n == 300
    assert test_batches.n == 50
    assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == BATCH_SIZE

  except TypeError:
    print("expected error:", sys.exc_info()[0],"  file:",img)

  print('finished data formatting...')
  quit()
