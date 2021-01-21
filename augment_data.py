#!/usr/bin/python3
# program to use tensorflow to augment data sets.
# initially tested on signed digits (0-9) in ~/Downloads/data/signed
# name: R. Melton
# date: 1/9/21

import os,shutil ,random
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
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from scipy import ndimage,misc
import itertools 
import os, sys 
import shutil 
import random, time 
import matplotlib.pyplot as pyplot
import imageio



def plots(ims,flgsize=(12,6),rows=1,interp=False,titles=None):
  if type(ims[0]) is np.ndarray:
    ims= np.array(ims)astype(np.uint8)
    if(ims.shape[-1] != 3):
      ims = ims.transpose((0,2,3,1))
  f = plt.figure(figsize=figsize)
  cols = len(ims)//rows if len(ims) %2 == 0 else len(ims)//rows+1
  for i in range(len(ims)):
    sp = r.add_subplot(rows,cols,i+1)
    sp.axis('Off')
    if titles is not None:
      sp.set_title(titles[i],fontsize=16)
    plt.imshow(ims[i],interpolation=None if interp else 'none')



#show picture on console of images
def plot_images(images_array):
  fig,axes = pyplot.subplots(1,10,figsize=(20,20))
  axes = axes.flatten()
  for img ,ax in zip(images_array,axes):
      ax.imshow(img)
      ax.axis('off')
  pyplot.tight_layout()
  pyplot.show()

def new_image_name(temp_dir_name):
  augname = ''
  number3 = random.randint(100,999)
  COMMON_NAME="IMG_A"  #A for augmented data and no file name colisions
  augname = COMMON_NAME+str(number3)
  print(augname)
  return augname

INPUT_DATA='/home/user01/Downloads/data/sd/Sign-Language-Digits-Dataset/Dataset'
OUTPUT_DATA='/home/user01/g/python3/data/sd/'
TRAIN_DATA='/home/user01/g/python3/data/sd/train'

TEMP_DIR='/home/user01/g/python3/temp'
#this is where the data currently lives on local disk
TRAIN_PATH=OUTPUT_DATA+'/train/'
#number to images to add to each digit directory (0-9)
NUM_AUGMENT=10
NUM_DIGITS=10

gen  = ImageDataGenerator(rotation_range=10,width_shift_range=0.1,
    height_shift_range=0.1,shear_range=0.15, zoom_range=0.1,
    channel_shift_range=10., horizontal_flip=True)
image_path = random.choice(os.listdir(OUTPUT_DATA+'train/0'))
print('old image name: ',old_image)
#https://www.youtube.com/watch?v=14syUbL16k4 at 1:40

if __name__ == '__main__':

  for digit in range (NUM_DIGITS):
    
    for img in range(NUM_AUGMENT):
      ran_image = random.choice(os.listdir(OUTPUT_DATA+f'train/{digit}'))
      full_path = OUTPUT_DATA+f'train/{digit}/{ran_image}'
      ran_image = np.expand_dims(imageio.imread(full_path),0)
      plt.imshow(image[0])

      print(f'{digit} random {ran_image}')
      source_dir = TRAIN_DATA+f'{digit}'
      new_name = new_image_name(' ')
      print(f'{digit} old image: {old_image}',f'{digit} new name: {new_name}')
      os.system(f"mkdir {TEMP_DIR}/{digit}")

      aug_iterator = generator.flow(ran_image)
      aug_images = [next(aug_iterator)[0].astype(np.uint8) for i in range(NUM_AUGMENT)]
      # for new_image in os.listdir(f'TEMP_DIR/{digit}')
      #   os.system(f"mv {TEMP_DIR}/{new_image}  {OUTPUT_DATA}/train/{i}/{new_image}")
      print(f'digit: {digit} aug list: {aug_images}')
    pyplot.plot(aug_images,figsize=(20,7),rows=2)


