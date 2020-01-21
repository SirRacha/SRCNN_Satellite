from model import SRCNN
from utils import (input_setup, create_required_directories)

import argparse
import numpy as np
import tensorflow as tf

import pprint
import os
import time
import sys


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("epoch", 15000, "Number of epoch [15000]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 21, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 21, "The size of stride to apply input image [12]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_boolean("build_model", None, "True for training, False for testing [True]")
flags.DEFINE_string("test_img", "/Test/Set5/bird_GT.bmp", "Add file relative to the project path to test.")
flags.DEFINE_string("temp_file_for_psnr", 'temp_file', "Temp file name when using psnr")
flags.DEFINE_string('bicubic_image', "bicubic_image.bmp", 'Bicubic file name')
# Required FLAG to run app
flags.mark_flag_as_required('build_model')


'''
######### alternative is argparse. Tensorflow 2 doesnt have Flags support. 

#def parse_args():
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', '-e', type=int, default=2)#, required=True)
parser.add_argument('--batch_size', '-bs', type=int, default=128)
parser.add_argument('--image_size', '-i', type=int, default=33)
parser.add_argument('--label_size', '-l', type=int, default=21)
parser.add_argument('--learning_rate', '-lr', type=int, default=1e-4)
parser.add_argument('--c_dim', '-cd', type=int, default=1)
parser.add_argument('--scale', '-sc', type=int, default=3)
parser.add_argument('--stride', '-s', type=int, default=21)
parser.add_argument('--checkpoint_dir', '-chk', type=str, default="checkpoint")
parser.add_argument('--sample_dir', '-sam', type=str, default="sample")
parser.add_argument('--build_model', '-bm', type=bool, default=True)
parser.add_argument('--test_img', '-testi', type=str, default="/Test/Set5/bird_GT.bmp")
parser.add_argument('--bicubic_image', '-bi', type=str, default="bicubic_image.bmp")
FLAGS = parser.parse_args()
'''

pp = pprint.PrettyPrinter()

def validate(img):
    img = img if img[:1]=='/' else ('/' + img)
    return os.getcwd() + img

def main(_):
  t0 = time.time()
    
  pp.pprint(FLAGS.build_model)
  
  if not FLAGS.build_model:
      FLAGS.test_img = validate(FLAGS.test_img)
      print("Image path = ", FLAGS.test_img)
      if not os.path.isfile(FLAGS.test_img):
          print("File does not exist ", FLAGS.test_img)
          sys.exit()

  create_required_directories(FLAGS)

  with tf.compat.v1.Session() as sess:
    srcnn = SRCNN(sess, 
                  image_size=FLAGS.image_size, 
                  label_size=FLAGS.label_size, 
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim, 
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir)

    if FLAGS.build_model:
      srcnn.train(FLAGS)
    else:
      srcnn.test(FLAGS)
  print("\n\nTime taken %4.2f\n\n" % (time.time() - t0))

# this stays outside of everything to run whole app when called
if __name__ == '__main__':
    #app.run(main)
    tf.compat.v1.app.run()

