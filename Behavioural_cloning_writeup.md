
# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/nvidia_arch.png
[image2]: ./writeup_images/Center_Driving.jpg
[image3]: ./writeup_images/Model_Summary.png
[image4]: ./writeup_images/Original_Image.jpg
[image5]: ./writeup_images/cropped_image.jpg
[image6]: ./writeup_images/flipped_image.jpg

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* behavioral_cloning_writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I have used a very well known architecture to build my model for cloning the behavior of the car. The architecture that I have used is NVIDIA architecture
with some modifications to train the neural network.
Below is the image of the actual NVIDIA architecture:

	NVIDIA ARCHITECTURE
	
 ![alt text][image1]
 
#### 2. Attempts to reduce overfitting in the model

To reduce the problem of overfitting I have used dropout on 2 convolutional layers and 1 fully connected layers.
The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I have used the training data set that was provided by Udacity as I was not able to drive properly and collect the data to train the model. so I decided to go with the data set provided and augment the data to generate more data so as to avoid overfitting the data and train the network better.  

Further details about the training are available in the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to design such an architecture that the vehicle stays in the center of the road.

My first step was to use a convolution neural network model similar to the NVIDIA architecture. I thought this model might be appropriate because the model is already designed for the same purpose.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set . I found that I had both a high training and validation loss which implied that the model was underfitting. Thus I decide to generate more data by augmenting the data by flipping the selective image sonly that had a steering angle greater than or less than 0. But this approach still did not seem to improve the model a lot so I decide to flip all the images in the data set irrespective of the steering angle.

Next after augmenting the data observed that I had a considerably low training loss but the validation loss is more. So, I was able derive a conclusion that I am overfitting the model. 

So to mitigate overfitting the model I have introduced dropout in the model in various layers of the model.   

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, so to improve the driving behavior in these cases, I changed the steering correction in the left and right.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes as mentioned below:

|Layer|Description|
|:-----------:|:-------------:|
|Cropping2D|crop the image fom tp and bottom|
|Lambda| Normalize the image in a range of 0-1|
|Convolution2D|5x5 kernel and an stride of 2x2|
|Activation | RELU|
|Dropout| Drop rate of 0.4| 
|Convolution2D|5x5 kernel and an stride of 2x2|
|Activation | RELU|
|Convolution2D|5x5 kernel and an stride of 2x2|
|Activation | RELU|
|Convolution2D|3x3 kernel and an stride of 1x1|
|Activation | RELU|
|Dropout| Drop rate of 0.4|
|Convolution2D|3x3 kernel and an stride of 1x1|
|Activation | RELU|
|Flatten| Flatten the layer|
|Fully Connected| FC layer of 100 neurons|
|Fully Connected| FC layer of 50 neurons|
|Dropout| Drop rate of 0.3|
|Fully Connected| FC layer of 10 neurons|
|Fully Connected| FC layer of 1 neuron(output)|

Model Summary:

![alt text][image3]

#### 3. Creation of the Training Set & Training Process

I have used the pre provided training data set and not created any data of my own as I was not sue on the driving skills that I have for driving a car on the computer. Here is an image of the center lane driving. 

![alt text][image2]

To augment the data sat, I also flipped images and angles as the data set provided is biased towards the left curves and the vehicle would tend to go towards the left with the data already provided. I have flipped the images using the numpy function fliplr and taking the negative value of the steering angles for the flipped images.
Below is a visualization of how the image look before and after flipping.
Original Image:

![alt text][image4]

Flipped Image:

![alt text][image6]

After the collection process, I had double the  number of data points that I initially had. I then preprocessed this data by cropping the images from top and bottom as these areas do not contribute much to the knowledge of the neural network on how to drive the car on road.
Below is a visualization of the images before and after cropping the image from top and bottom.
Original Image:

![alt text][image4]

Cropped Image:

![alt text][image5]

Then I normalized the images so that the pixel values of the image all lie in a range of 0-1.

I finally randomly shuffled the data set and put 20% of the data into a validation set. and used the 80% data set to train the model to drive a car autonomously. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by a training loss of .022 and a validation loss of.019. I used an Adam optimizer so that manually training the learning rate wasn't necessary.
