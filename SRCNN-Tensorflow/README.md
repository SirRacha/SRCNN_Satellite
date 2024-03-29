# SRCNN-Tensorflow
Tensorflow implementation of Convolutional Neural Networks for super-resolution. The original Matlab and Caffe from official website can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html).

## Local Dev

### Requirements
 * virtualenv
 * Tensorflow 1.6
 * Scipy version = 1.2.2
 * h5py
 * matplotlib

This code requires Tensorflow and Scipy 1.2.2 because of the imread deprication. 

### Notes on unique aspects of this code.
    * added in PSNR metrics for every run
    * changeed optimizer to Adam, or SVG with momentum depending on error assessment =
        * added Adam on train and test functions and model performs faster.
        * Working on how to add more accuracy metrics on performance
    * cannot change graycsale easily. Upon more reading, the source paper states isolating the Y channel in YCbCr performs          best. Is why grayscale has its tendrils throughout the image processing and models
    * Keras is a high-level tool that does not allow for the exact fine-tuning of hyperparameters that SRCNN needs.
    * PyTorch may be a low-level alternative to rewrite the code in.
    * The paper states that increasing number of input images does not equal better performance since the patches made in subimage processing creates such a large number of real examples. (91 images = 24800 subimages)
    * increased stride numbers.
        * stride is set at 21.   mask is 33x33


### Create Environment
(One time) Create virtual environment

```bash
$ virtualenv env
```

Activate virtual environment. **Note, all Python commands below assume the virtual environment is activated.**

```bash
$ source env/bin/activate
```

Add project root directory to Python path

```bash
(env) $ export PYTHONPATH=$PWD
```

Install dependencies. Repeat whenever dependencies are added.  
Note: to create a dependancy file use   ```pip freeze > requirements.txt```

```bash
(env) $ pip3 install -r requirements.txt
```

## Usage
For training, `python main.py`  and set any tf.flags (ie hyperpameters with the -- prefix
<br>
For testing, `$  python main.py --build_model=False --test_img=/Test/Set5/butterfly_GT.bmp`

## Result


## References
* [tegg89/SRCNN-Tensorflow](https://github.com/tegg89/SRCNN-Tensorflow)
  * - I built this off of this repository. Also, updated code to reflect new TensorFlow and Python packages/libraries.
<br>

* [liliumao/Tensorflow-srcnn](https://github.com/liliumao/Tensorflow-srcnn) 
  * - I referred to this repository which is same implementation using Matlab code and Caffe model.
<br>

* [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) 
  * - I have followed and learned training process and structure of this repository.

