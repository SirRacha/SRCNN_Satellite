"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import random
import matplotlib.pyplot as plt

#from PIL import Image  # for loading images as YCbCr format
from PIL import Image
import argparse
import scipy.misc
import scipy.ndimage
import numpy as np

import tensorflow as tf

try:
  xrange
except:
  xrange = range
  
FLAGS = tf.app.flags.FLAGS

def read_data(path):
  """
  Read h5 format data file
  
  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
  with h5py.File(path, 'r') as hf:
    data = np.array(hf.get('data'))
    label = np.array(hf.get('label'))
    return data, label
  
# use imageio.imread instead ?  YCbCr color video capability is in PIL library #
# the paper states isolating Y channel for train/test returns = best result and leaving CbCr channels as bicubic interpolation upscaling.
# alternatively, using RGB is also possible to get good returns 
def preprocess(path, config):
  """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation
  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
  image = imread(path, is_grayscale=True)
  label_ = modcrop(image, config.scale)
  
  # Must be normalized. sets both =1 float. 255 is grayscale. black =0, white=1. makes a flat image. a vector
  image = image / 255.
  label_ = label_ / 255.
  
# filling in the gaps with interpolation.
  #- checked and this has to be referenced as it is.  will keep as input0_. why is it structured this way?
  input0_ = scipy.ndimage.interpolation.zoom(label_, (1./config.scale), prefilter=False)
  input_ = scipy.ndimage.interpolation.zoom(input0_, (config.scale/1.), prefilter=False)
  imsave(input_, get_bicubic_file_path(config))
  return input_, label_


def get_filepaths(sess, dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  if FLAGS.build_model:
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
  else:
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
    data = glob.glob(os.path.join(data_dir, "*.bmp"))

  return data

def persist_data(data, label):
  """
  Make input data as h5 file format
  Depending on 'is_train' NOW build_model FLAG (flag value), savepath would be changed.
  """
  if FLAGS.build_model:
    savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
  else:
    savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data', data=data)
    hf.create_dataset('label', data=label)

def imread(path, is_grayscale=True):
  """
  Read image using its path.
  The flatten parameter makes it grayscale. in imageio this is "is_gray=" -- but when i make it False, it wont creat size of the image to 33x33
  """
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def input_setup(sess, config):
  """
  Read image files and make their sub-images and saved them as a h5 file format.
  """
  # Load data path
  # gets every image with bmp into this variable. 91 total
  # here we resize the image; part of Convolution part 1
 
  temp_file=None
  if config.build_model:
    # TRAIN DATASET
    data = get_filepaths(sess, dataset="Train")
  else:
    # TEST DATASET
    data = config.test_img 

  sub_input_sequence = []
  sub_label_sequence = []
  
  # the paper states to have No padding bc it will mess up the subimages split, train, and merging
  # padding = abs(config.image_size - config.label_size) / 2 # 6
  padding = 0 #testED to see if boxes occur. THEY DONT
    
###### FOR TRAINNG  #######
  if config.build_model:
    for i in xrange(len(data)):
      input_, label_ = preprocess(data[i], config)

      if len(input_.shape) == 3:
        h, w, _ = input_.shape
      else:
        h, w = input_.shape 

        ############ creating 2 different vectors for the original and upscaled chips/patches (that are 9x9) ########
      for x in range(0, h-config.image_size+1, config.stride):
        for y in range(0, w-config.image_size+1, config.stride):
          sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]  - this goes through image with a Mask
          sub_label = label_[x+int(padding):x+int(padding)+config.label_size, y+int(padding):y+int(padding)+config.label_size] # [21 x 21]

          # Make channel value ## in a 3D plane
          sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
          sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

          sub_input_sequence.append(sub_input)
          sub_label_sequence.append(sub_label)
          
    print(" Sub-Input number of labels is ", len(sub_label_sequence))
          
####### FOR TEST  ##########
  else:
    input_, label_ = preprocess(data, config)

    if len(input_.shape) == 3:
      h, w, _ = input_.shape
    else:
      h, w = input_.shape

    # Numbers of sub-images in height and width of image are needed to compute merge operation.
    nx = ny = 0 
    for x in range(0, h-config.image_size+1, config.stride):
      nx += 1; ny = 0
      for y in range(0, w-config.image_size+1, config.stride):
        ny += 1
        sub_input = input_[x:x+config.image_size, y:y+config.image_size] # [33 x 33]
        sub_label = label_[x+int(padding):x+int(padding)+config.label_size, y+int(padding):y+int(padding)+config.label_size] # [21 x 21]
        
        sub_input = sub_input.reshape([config.image_size, config.image_size, 1])  
        sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

        sub_input_sequence.append(sub_input)
        sub_label_sequence.append(sub_label)

    print(" Sub-Input number of images is ", len(sub_input_sequence))
    print(" Sub-Input number of labels is ", len(sub_label_sequence))
  """
  len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
  (sub_input_sequence[0]).shape : (33, 33, 1)
  """
  # Make list to numpy array. With this transform
  arrdata = np.asarray(sub_input_sequence) # [?, 33, 33, 1]
  arrlabel = np.asarray(sub_label_sequence) # [?, 21, 21, 1]

  persist_data(arrdata, arrlabel)
  
#### can we remove this because of the else statement above?#########  NOOOOOOOOO
  if not config.build_model:
    return nx, ny
 
#saving final image ###  NEED TO CHANGE THIS TO BE MORE GLOBAL TO SAVE THE BICUBIC IMAGE TOO
def imsave(image, path):
  #print("Saving " + path)
  return scipy.misc.imsave(path, image)

#the subimages/patches are merged back together after upscaling###
def merge(images, size):
  print("Merge occurring for " + str(len(images)) + " images")
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h*size[0], w*size[1], 1))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img

#image paths will be defined in model.py 
def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator
  
def psnr(im1, im2):
    
  print('psnr called with {} and {}'.format(im1, im2))
  x = imread(im1, is_grayscale=True)
  print("Size x is " + str(len(x)))
  y = imread(im2, is_grayscale=True)
  print("Size y is " + str(len(y)))
  
  img_arr1 = np.array(x).astype('float32')
  img_arr2 = np.array(y).astype('float32')
  print(img_arr1.shape)
  
  # The dimensions were not the same, sooooo we need to make these the same size
  # Convert np array to PIL object for cropping to 288x288
  img_pil = Image.fromarray(img_arr1)
  area = (0, 0, 288, 288)
  img_crop = img_pil.crop(area)
  # Convert back to np array (288,288)
  img_arr1 = np.array(img_crop)
  print ("cropped size is", img_arr1.shape)

  mse = tf.reduce_mean(tf.math.squared_difference(img_arr1, img_arr2))
  psnr = tf.constant(255**2, dtype=tf.float32)/mse
  result = tf.constant(10, dtype=tf.float32)*log10(psnr)
  with tf.compat.v1.Session():
    result = result.eval()
    return result
  print("Computing PSNR...")
  
def get_orig_file_path(config):
  return os.path.join(os.getcwd(), config.sample_dir, config.test_img)
  
def get_bicubic_file_path(config):
  return os.path.join(os.getcwd(), config.sample_dir, config.bicubic_image)
  
def create_required_directories(config):
  if not os.path.exists(config.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(config.sample_dir):
    os.makedirs(FLAGS.sample_dir)    
