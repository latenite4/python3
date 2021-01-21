#!/usr/bin/python3

import os,zipfile
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
import shutil ,distro
import random, time 
import matplotlib.pyplot as pyplot
import imageio,platform

def show_versions_info():
  print('local software versions:')
  print("keras version: ",keras.__version__)
  print("tensorflow version: ",tf.__version__)
  print(f'numpy version: {np.__version__}')
  print(f"python version: {platform.python_version()}")
  print("Cuda version: ")
  os.system('nvcc --version')
  print(f"Linux distribution: {distro.linux_distribution()}")
  print(f'Operating system kernel: {platform.platform()}\n')
