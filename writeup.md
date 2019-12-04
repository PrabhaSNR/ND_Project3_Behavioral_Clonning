#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[center_straight_line]: ./output/Centre_Lane.jpg "Center Straight Line"
[center_curve]: ./output/Curve_Centre_Lane.jpg "Center Curve"
[left]: ./output/LeftCamera.jpg "Left Camera"
[right]: ./output/RightCamera.jpg "Right Camera"
[center_recovering]: ./output/CentreRecovering.jpg "Center Recovering"
[left_recovering]: ./output/Left_Recovering.jpg "Left Recovering"
[right_recovering]: ./output/RightRecovering.jpg "Right Recovering"
[training_validation_loss]: ./output/Figure_1.png "Training and Validation losses"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network 
* **writeup.md** summarizing the results

####2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing the following command:

```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model used in my project is based on the Nvidia DNN . It shows better performance than the LeNet-5 model that I used before. 

It consists of three 5x5 convolution layers, two 3x3 convolution layers, and three fully-connected layers (model.py line 57 to 66). It also includes images normalization as described in the paper, using a Keras Lambda layer (model.py line 55). In addition to that, I've cropped the images to use the same shape as in the paper (model.py line 56).

The Adam optimizer was chosen with the default parameters to optimize the mean squared error (MSE) loss function. The final output layer is a simple fully-connected layer with a single neuron (model.py line 66).

####2. Attempts to reduce overfitting in the model

No regularization has been added to the layer since the predictions where were good. 

Moreover, the model was trained on data properly collected to ensure the car is not biased toward certains steering angles. See section **'4. Appropriate training data'** for more details.

####3. Model parameter tuning

As mention before, the model use an adam optimizer with a default learning rate equal to 0.001. 

It was trained during 5 epochs only because after that the loss is not getting much better. 

The batch size was kept to a minimum of 32 samples. 

The data was splitted between training and validation data using a rate of 0.2 for the validation set.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road as close to the center as possible. I trained it for three laps. In addition to that, I drove the car for one lap counter-clockwise to avoid the data being biased towards left turns. It also tends to reduce overfitting and helps the model generalizes better.

Finally, I've used recovery driving from the sides where the car was going off track during test sessions. It helped the model learn how to drive smoothly around curves.


###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use an iterative proccess. I've switched very often between tuning/training and testing on the track. It helped me a lot to understand how well or bad my changes was helping the model.

My first step was to use a well-kwown convolution neural network model, LeNet-5. This choice was made because this model is simple to implement and quick to train. However, it was not performing very well on this task. So, I've implemented the model introduced by Nvidia which performed much better for driving the car autonomously. It has more convolution layers than LeNet-5.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

Here's the visualization of the losses after each epoch:

![Training and Validation losses][training_validation_loss]

There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I've recorded recovering data from the sides back to the center at these spots.

Between each change I run the simulator to see how well the car was driving around track one. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture has been described in section **'1. An appropriate model architecture has been employed'**.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here are example images of center lane driving:

![Center lane driving][center_straight_line]
![Curve lane driving][center_curve]

I then recorded the vehicle recovering from the sides where the car was going off track back to center so that the vehicle would learn how to handle these situations, especially sharp curves. These images show what a recovery looks like starting where the car was going off the road (brown border on the right):

![Center recovering][center_recovering]
![Left recovering][left_recovering]
![Right recovering][right_recovering]

I haven't used track two because the model successfully pass the first track without any other data. As a future improvement I'll augment the data by recording on track two to improve the model.

However, to augment the data set, I've used left and rights images and applied a correction of 0.20. This value is emperical but it has proven to work. Furthermore, I've flipped images and and I've took the opposite sign of the steering measurement. It helps reduce overfitting by augmenting data even more without taking too much time on data collection.

Here are example images from left and right cameras:

![Left camera][left]
![Right camera][right]

I finally randomly shuffled the data set and put 20% of the data into a validation set.  

I've cropped every images to remove the uneccesary part (trees, sky, car front...) and resized them to feed the model with the right shape as described by Nvidia.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 because the loss wasn't improved beyond that point. I used an adam optimizer with a default learning rate (0.001) from the Keras library.