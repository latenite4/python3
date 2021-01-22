#!/usr/bin/python3
# util functions to use with pyplot
# name: R. Melton
# date: 1/21/2021
import sys
import matplotlib.pyplot as plt

# create callback for timer
def close_event():
  plt.close() 
  sys.exit(0)

#create timer which will remove image show()
def make_timer(time_sec,fig):
  timer = fig.canvas.new_timer(interval = int(time_sec)*1000) #creating a timer object and setting an interval of 3000 milliseconds
  timer.add_callback(close_event)
  return timer
