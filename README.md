[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
### Overview

### Udacity-Traffic-Sign-Classifier-Project
  This project aim to using the knowledge learned from Udacity Self-driving Car nano degree coourse to build a CNN architecture to 
classify the traffic sign.

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Writeup / README

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![](bar%20chart%20of%20dataset.png)

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because, after visiualize the dataset, I found the color of the traffic signs are various, and since the shape and letters of the traffic signs are way more important than the color. Since the grayscale image is monochrome and only contains the density information. So I decide to use grayscale image to remove 'noise' information of dataset.

As a last step, I normalized the image data because the data norlization will prevent overfitting. Ohter than this, the image data will various from 0 to 255 instead of 0 to 1. In this project, I use the feature scaling method. Or, I can use the scikit-learn built in z-score method (not this time).

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution layer1 3x3     	| 1x1 stride, valid padding, outputs 28x28x20 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x40 				|
| Convolution layer2 3x3	    | 1x1 stride, valid padding, outputs 12x12x80    |
| RELU          |                       |
| Convolution layer3 3x3	    | 1x1 stride, valid padding, outputs 10x10x80     |
| RELU          |     |
| Max pooling	      	| 2x2 stride,  outputs 5x5x80 				|
| Fully connected		| Input 2000, output 120        									|
|RELU                  |                                    |
| Fully connected		| Input 120, output 84        									|
|RELU                   |                                   |
| Fully connected		| Input 84, output 43        									|
|RELU                    |                                  |
| Softmax				| Cross_entropy       									|
|Optimizer						|Adam algorithm												|



#### 3. Describe how you trained your model.

To train the model, I used an architecture based on LeNet. The batch size is 128 and epochs is 20. I do the 10 epochs first, and I found the accuracy will increase a little if I add more epochs, so I change the epochs to 20. However, if I keep increasing the epochs to 30 or higher, the accuracy will not go higher. 

Also, I add one more Cnn layer to the LeNet model to make the model deeper to get higher accuracy. The original LeNet model gave me 89% accuracy, after add one more layer, it gives me 94% accuracy.

Thirdly, to prevent overfitting, I use the dropout layer and the accuracy goes up.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are right German traffic signs that I found on the web:

![alt text](/GTS/1.jpg)![alt text](/GTS/2.jpg)![alt text](/GTS/3.jpg)![alt text](/GTS/4.jpg)
![alt text](/GTS/5.jpg)![alt text](/GTS/6.jpg)![alt text](/GTS/7.jpg)![alt text](/GTS/8.jpg)

The seventh image might be difficult to classify because the training data of this image is less than the others

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Right-of-way at the next intersection     			| Right-of-way at the next intersection 										|
| Speed limit (70km/h)					| Speed limit (70km/h)											|
| Roundabout mandatory	      		| Speed limit (60km/h)					 				|
| Vehicles over 3.5 metric tons prohibited		| Vehicles over 3.5 metric tons prohibited      							|
| No Pass    | Wild animals crossing|
| Speed limit (30km/h) | Speed limit (30km/h)| 


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

*Output: Image 0 probabilities: [  9.99999881e-01   7.39858663e-08   1.18372752e-08   1.26846172e-10 6.91618707e-11] and predicted classes: [14  5  1  6  3]

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop sign   									| 
| .0001     				| Speed limit (80km/h) 										|
| .0001					    | Speed limit (30km/h)											|
| .0001	      			| End of speed limit (80km/h)					 				|
| .0001				      | Speed limit (60km/h)      							|

### Files Submitted
   GTS: Folder contains the test image to test the accuracy of this CNN architecture
   Signnames.csv: Labels of each traffic signs
   train.p: dataset used to train the model
   valid.p: dataset used to valid the model
   test.p: dataset used to test the model
   Traffic_Sign_Classifier.ipynb: Code
   README.md

### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/) 

### Dataset

1. [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32.
2. Clone the project and start the notebook.
```
git clone https://github.com/udacity/CarND-Traffic-Signs
cd CarND-Traffic-Signs
jupyter notebook Traffic_Signs_Recognition.ipynb
```
3. Follow the instructions in the `Traffic_Signs_Recognition.ipynb` notebook.
