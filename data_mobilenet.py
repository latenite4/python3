#!/usr/bin/python3

#this program is only run by the try.sh script.

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
INPUT_DATA='/home/user01/Downloads/data/sd/Sign-Language-Digits-Dataset/Dataset'
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

if __name__ == '__main__':
  try:
    if os.path.exists(OUTPUT_DATA+'/0') is False:
      print('reformatting data for mobilenet')
      os.chdir(OUTPUT_DATA)


      for i in range(0,10):
        print(f'\nreformatting data digit: {i}')
        # make sample input for validity testing
        print(f'source dir: {INPUT_DATA}/{i}/')
        valid_samples = random.sample(os.listdir(f'{INPUT_DATA}/{i}/'),30)
        print(f'>valid samples for: {i} {valid_samples}')
        for v in valid_samples:
          os.system(f"mv {INPUT_DATA}/{i}/{v}  {OUTPUT_DATA}/valid/{i}")
        
        test_samples = random.sample(os.listdir(f'{INPUT_DATA}/{i}/'),5)
        print(f'>>test samples for: {i} {test_samples}')
        for t in test_samples:
          os.system(f"mv {INPUT_DATA}/{i}/{t}  {OUTPUT_DATA}/test/{i}")
        
        train_samples = os.listdir(f'{INPUT_DATA}/{i}/')
        print(f'>>>train samples for: {i} {train_samples}')
      
        #the shutil.move() python function does not work
        for tr in train_samples:
          os.system(f"mv {INPUT_DATA}/{i}/{tr}  {OUTPUT_DATA}/train/{i}")

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
    print("expected error:", sys.exc_info()[0])

  print('finished data formatting...')
  quit()
