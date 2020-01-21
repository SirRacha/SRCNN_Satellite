from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  psnr,
  get_orig_file_path
)

import time
import os

import numpy as np
import tensorflow as tf

try:
  xrange
except:
  xrange = range

#### The SRCNN class including Train and Test functions.  ####
class SRCNN(object):

  def __init__(self, 
               sess, 
               image_size=33,
               label_size=21, 
               batch_size=128,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    
#remove the self.build_model() from this class and just call from main method. 
#This negates need to say True/False when calling main.py
    self.build_model()

  #  keep this here. it sets up the model as empty. 
  #  This combined with self.model below actually creates the CNN
  def build_model(self):
    print("creating TF placeholder 'images' of size ", (self.image_size, self.image_size, self.c_dim))
    print("creating TF placeholder 'labels' of size ", (self.label_size, self.label_size, self.c_dim))
    self.images = tf.compat.v1.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
    self.labels = tf.compat.v1.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
  
  ####  Weights as defined in paper are 9-1-5 filter with n1=64 and n2=32 convolutions ####
    self.weights = {
      'w1': tf.Variable(tf.random.normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
      'w2': tf.Variable(tf.random.normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
      'w3': tf.Variable(tf.random.normal([5, 5, 32, 1], stddev=1e-3), name='w3')
    }
    self.biases = {
      'b1': tf.Variable(tf.zeros([64]), name='b1'),
      'b2': tf.Variable(tf.zeros([32]), name='b2'),
      'b3': tf.Variable(tf.zeros([1]), name='b3')
    }

 #  this is where the model runs. the actual training or test of model ### this will fill the buckets. model see line 130
    self.pred = self.model()

#Loss function (Mean Squared Error, MSE) as stated in paper. MSE is best for Gaussian (normal) distribution of regression problems.
#Recommend using cross entropy loss instead of MSE if going for classification problems and Bernoulli distribution. 
#there is also tf.reduce_sum() that performs MSE with more detailed parameters.
    self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
    
 # create logger with Saver function. fill with tf checkpoints into folder
    self.saver = tf.compat.v1.train.Saver()

#### TRAIN function ####
  def train(self, config):
    if config.build_model:
      input_setup(self.sess, config)
    else:
      nx, ny = input_setup(self.sess, config)
   
    data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")

    train_data, train_label = read_data(data_dir)

    # Paper suggests Stochastic Gradient Descent -SGD- with the standard backpropagation. But now changed to Adam with standard backpropagation
    # bc Adam is faster is works similarly. Another alternative would be Stochastic gradient descent (SGD) with momentum.
    self.train_op = tf.compat.v1.train.AdamOptimizer(config.learning_rate).minimize(self.loss)
 
    # Initialize the TF variables defined above (e.g. b1, b2, etc.)
    # tf.compat.v1.global_variables_initializer.run()  --- warnings say this may be depreicated too
    #    tf.initialize_all_variables
    tf.compat.v1.global_variables_initializer().run()

    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
      print("Training...")
      
#  train data is set size and batch size is a set size. they are divided by integer below. 
      for ep in xrange(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data) // config.batch_size
        print("loop len is equal to:  ", batch_idxs) # indicates how long will run
        for idx in xrange(0, batch_idxs):
          batch_images = train_data[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels = train_label[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          
          # this runs the model for train and gets error loss
          _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
          # this prints the metrics for eopch evaluation
          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err))

          if counter % 500 == 0:
            self.save(config.checkpoint_dir, counter)

####  TEST function  ####
  def test(self, config):
    nx, ny = input_setup(self.sess, config)

    data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

    train_data, train_label = read_data(data_dir)
    
    # Adam Optimizer - changed from paper that uses GradientDescent
    self.train_op = tf.compat.v1.train.AdamOptimizer(config.learning_rate).minimize(self.loss)

    # Initialize the TF variables that are defined (e.g. b1, b2, etc.)
    tf.compat.v1.global_variables_initializer().run()
    
    if self.load(self.checkpoint_dir):
      print(" [*] Load Checkpoint SUCCESS")
    else:
      print(" [!] Load failed...")
      
    result = self.pred.eval({self.images: train_data, self.labels: train_label})
    print("Calling merge...")
    result = merge(result, [nx, ny])
    result = result.squeeze()
    result_image_path = os.path.join(os.getcwd(), config.sample_dir)
    result_image_path = os.path.join(result_image_path, "test_image.png")
    print("Saving file to " + result_image_path)
    imsave(result, result_image_path)
    print("image SAVED")
    
    # Get the temp image that was originally saved
    temp_path = get_orig_file_path(config)
    
    # SAVE BICUBIC INTERPOLATION IMG
    print("Saving file to " + temp_path)
    print("bicubic SAVED")

    print("The PSNR for final image against the original image is: ", psnr(result_image_path, temp_path))
    
#  creates the model using the parameters (weights/biases).  stride == 21.
#  2 relu conv2d input, and 1 conv2d for output
  def model(self):
    print("Created model with conv2d, relu, conv2d, relu, conv2d")
    conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID') + self.biases['b1'])
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID') + self.biases['b2'])
    conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='VALID') + self.biases['b3']
    return conv3

  def save(self, checkpoint_dir, step):
    model_name = "SRCNN.model"
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("srcnn", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False