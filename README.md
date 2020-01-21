
## These directories contain different approaches to working with SRCNN. 
## They include nearly complete submissions for Super Resolution Convolutional Neural Network (SRCNN) algorithm.


###    SRCNN-Tensorflow is our main focus for completing this application. 
<br>

###    *SRCNN-COLAB* showcases how to go about setting up Tensorflow models in the Google Colab environment.
<br>

###    *SRCNN-Keras*  is a nonworking proof of concept.
<br>


## Model Architecture
### Super Resolution CNN (SRCNN)
<img src="https://raw.githubusercontent.com/titu1994/ImageSuperResolution/master/architectures/SRCNN.png" height=100% width=25%>

The model above is take from user *titu1994* who I cite as a source for this project. 

I have some differences from the original paper, and from other scientists who have worked on this topic:
<br><b>[1]</b> Used the Adam optimizer instead of Stochastic Gradient Descent.
<br><b>[2]</b> Stride is set to 21 instead 14.

My models underperform compared to the results posted in the paper. I was unable to replicate the GPU power and time for running to get my PSNR as high, ie. 32dbs.

I have optimized this model for use on different imagery instead of the usual 91 images from ImageNET.

