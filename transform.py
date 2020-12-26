
from image import Image
import numpy as np
import time

def adjust_brightness(image,factor):
  #scale each value by some amount
  x_pixels, y_pixels,num_channels = image.array.shape
  new_im = Image(x_pixels=x_pixels,y_pixels=y_pixels,num_channels=num_channels)
  for x in range(x_pixels):
    for y in range(y_pixels):
      for c in range(num_channels):
        new_im.array[x,y,c] = image.array[x,y,c] * factor  #non vectorized version
  
  #vectorized version
  # new_im.array = image.array * factor -# this is faster
  
  return new_im

#adjust the contrast by increasing difference from user
#defined midpoint
def adjust_contrast(image, factor, mid=0.5):
  x_pixels, y_pixels,num_channels = image.array.shape
  new_im = Image(x_pixels=x_pixels,y_pixels=y_pixels,num_channels=num_channels)
  for x in range(x_pixels):
      for y in range(y_pixels):
        for c in range(num_channels):
          new_im.array[x,y,c] = (image.array[x,y,c] -mid)* factor + mid  #non vectorized version
  
  #vectorized version
  # new_im.array = (image.array - mid) * factor + mid

  return new_im
def blur(image, k_size):
  #k_size is the number of pixels to use when doing the blur
  #k_size=3 would be above and below and left neighbor, right neighbor pixels, and diagonal
  #neighbor pixels.
  x_pixels, y_pixels,num_channels = image.array.shape
  new_im = Image(x_pixels=x_pixels,y_pixels=y_pixels,num_channels=num_channels)
  neighbor_range = k_size // 2
 
  for x in range(x_pixels):
    for y in range(y_pixels):
      for c in range(num_channels):
        total = 0
        for x_i in range(max(0,x-neighbor_range), min(new_im.x_pixels-1, x+neighbor_range)+1):
          for y_i in range(max(0,y-neighbor_range), min(new_im.y_pixels-1, y+neighbor_range)+1):
            total += image.array[x_i, y_i, c]
        new_im.array[x,y,c] = total / (k_size **2)  # average for kernel size in image

  return new_im

def apply_kernel(image, kernel):
    # the kernel should be a 2D array that represents the kernel we'll use!
    # for the sake of simiplicity of this implementation, let's assume that the kernel is SQUARE
    # for example the sobel x kernel (detecting horizontal edges) is as follows:
    # [1 0 -1]
    # [2 0 -2]
    # [1 0 -1]
    x_pixels, y_pixels, num_channels = image.array.shape  # represents x, y pixels of image, # channels (R, G, B)
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  # making a new array to copy values to!
    neighbor_range = kernel.shape[0] // 2  # this is a variable that tells us how many neighbors we actually look at (ie for a 3x3 kernel, this value should be 1)
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                total = 0
                for x_i in range(max(0,x-neighbor_range), min(new_im.x_pixels-1, x+neighbor_range)+1):
                    for y_i in range(max(0,y-neighbor_range), min(new_im.y_pixels-1, y+neighbor_range)+1):
                        x_k = x_i + neighbor_range - x
                        y_k = y_i + neighbor_range - y
                        kernel_val = kernel[x_k, y_k]
                        total += image.array[x_i, y_i, c] * kernel_val
                new_im.array[x, y, c] = total
    return new_im


def combine_images(image1, image2):
    # let's combine two images using the squared sum of squares: value = sqrt(value_1**2, value_2**2)
    # size of image1 and image2 MUST be the same
    x_pixels, y_pixels, num_channels = image1.array.shape  # represents x, y pixels of image, # channels (R, G, B)
    new_im = Image(x_pixels=x_pixels, y_pixels=y_pixels, num_channels=num_channels)  # making a new array to copy values to!
    for x in range(x_pixels):
        for y in range(y_pixels):
            for c in range(num_channels):
                new_im.array[x, y, c] = (image1.array[x, y, c]**2 + image2.array[x, y, c]**2)**0.5
    return new_im

if __name__ == '__main__':
  lake = Image(filename = 'lake.png')
  city = Image(filename='city.png')
  start_time = time.time()

  # brightened_im = adjust_brightness(lake, 1.7)
  # brightened_im.write_image('brightened.png')
  # darkened_im = adjust_brightness(lake, 0.3)
  # darkened_im.write_image('darkened.png')

  # incr_contrast = adjust_contrast(lake, 2,0.5)
  # incr_contrast.write_image('incr_contrast.png')
  # decr_contrast = adjust_contrast(lake, 0.5,0.5)
  # decr_contrast.write_image('decr_contrast.png')

  # blur_3 = blur(city,3)
  # blur_3.write_image('blur_k3.png') 
  # blur_15 = blur(city,15)
  # blur_15.write_image('blur_k15.png') 


# let's apply a sobel kernel on the x and y axis
  sobel_x = apply_kernel(city, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
  sobel_x.write_image('edge_x.png')
  sobel_y = apply_kernel(city, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
  sobel_y.write_image('edge_y.png')

  # this will show x and y edges
  sobel_xy = combine_images(sobel_x, sobel_y)
  sobel_xy.write_image('edge_xy.png')

  print(f'total execution duration: {time.time() - start_time}s')