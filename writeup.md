**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

[//]: # (Image References)

[image1]: ./example_center.jpg "Normal image"

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* run.mp4 contains a recording of driving in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can in principal be driven autonomously around the track by executing 

```
python drive.py model.h5
```

However, the code contains a fix for systems which use commas as decimal separator on
lines 51-52. Change these lines to use the code on a system with English locale.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 89-93). Above the convolution network are
four fully connected layers (code line 95-98).

The model includes RELU layers to introduce nonlinearity (code line 89-93), and the data is normalized in the model using a Keras lambda layer (code line 86). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets (code line 103-107). It turned out that the model started to overfit after 3 epochs. For this reason I stopped training after 3 epochs (code line 107).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code line line 101).

#### 4 . Appropriate training data

Training data was acquired by driving several laps on track 1 in both directions.  I wanted to try how the model would train with "normal" driving data so I decided not to acquire any recovery data but a rather large amount of conventional driving. Furthermore I selected only one out of five subsequent frames  to reduce the amount of data. It turned out that this data was sufficient.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a given architecture and optimize the result by acquiring more training data.

My first step was to use a convolution neural network model similar to the NVIDIA architecture which has been proven to be suitable for similar problems.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. To combat the overfitting, I stopped the training as soon as the validation error started to rise.

The next step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I improved the driving behavior by augmenting the training data. I flipped each training image (code line 14) and used the left and right extra cameras (code line 13) when setting up the training data.

Then I acquired extra training data to improve the result further.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 85-98) consisted of a convolution neural network with the following layers and layer sizes: 

| Layer         	|     Description	        		| 
|:---------------------:|:---------------------------------------------:| 
| Input         	| 160x320x3 Color image   			|
| Cropping         	| remove top 70 and lower 25 lines       	|
| Lambda	      	| scale color channel to [-0.5, 0.5]		|
| Convolution   	| 2x2 stride, 5x5 kernel size, depth 24         |
| RELU			|						|
| Convolution   	| 2x2 stride, 5x5 kernel size, depth 36         |
| RELU			|						|
| Convolution   	| 2x2 stride, 5x5 kernel size, depth 48         |
| RELU			|						|
| Convolution   	| 1x1 stride, 3x3 kernel size, depth 64         |
| RELU			|						|
| Convolution   	| 1x1 stride, 3x3 kernel size, depth 64         |
| RELU			|						|
| Flatten          	|                                               |
| Fully connected	| depth 100    				        |
| Fully connected	| depth 50    				        |
| Fully connected	| depth 10    				        |
| Fully connected	| depth 10    				        |

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded several laps on track one in each direction using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

After the collection process, I had 66762 images (including left and right images). Then I flipped each image which resulted in about 130000 data points. For training I used only one frame out five, i.e. about 26000 data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped to determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by training the model with more epochs and looking at the performance of the training set versus the validation set. I used an adam optimizer so that manually training the learning rate wasn't necessary.
