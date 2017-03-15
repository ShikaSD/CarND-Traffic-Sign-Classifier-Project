# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/chart1.png "Visualization"
[image2]: ./img/image_layers.png "Image with layers"
[image3]: ./img/image_augmented.png "Augmented image"
[image4]: ./custom_signs/1.jpg "Traffic Sign 1"
[image5]: ./custom_signs/10.jpg "Traffic Sign 2"
[image6]: ./custom_signs/14.jpg "Traffic Sign 3"
[image7]: ./custom_signs/20.jpg "Traffic Sign 4"
[image8]: ./custom_signs/21.png "Traffic Sign 5"
[image9]: ./custom_signs/25.jpg "Traffic Sign 6"
[image10]: ./custom_signs/26.jpg "Traffic Sign 7"
[image11]: ./custom_signs/28.jpg "Traffic Sign 8"
[image12]: ./custom_signs/35.jpg "Traffic Sign 9"
[image13]: ./custom_signs/41.jpg "Traffic Sign 10"
[image14]: ./img/chart2.png "Softmax"
[image15]: ./img/feature_maps.png "feature_maps"

---
Link to my [project code](https://github.com/ShikaSD/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb), [exported html](https://rawgit.com/ShikaSD/CarND-Traffic-Sign-Classifier-Project/master/report.html).

### Data Set Summary & Exploration

#### 1. Basic summary of the dataset

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas and numpy library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of unique labels = 43

#### 2. Visualization

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data spread into classes

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing

As a first step, I decided to convert the images to LAB space to normalize brightness using application of CLAHE algorithm along the L canal.

Then, the image is converted back to the RGB space and grayscale, red and blue channels are taken to feed inside.

Here is an example of a traffic sign image before and after preprocessing.

![alt text][image2]

As a last step, I normalized the image data to ease its data understanding for the NN.

#### 2. Data setup

The input data was already split to the sets needed.

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.

I decided to generate additional data to make NN understand more general concept of the sign, not to remember it shape and relative position of the features. To add more data to the the data set, I used the following techniques: rotation to ±15 degrees, affine transform with deviation of 10 pixels and image blur to smooth towards general features.

Here is an example of an original image and an augmented image:

![alt text][image3]

The augmentation rate was decreasing over the epochs for better fit of the model. Finally, (the results are below) the NN trained on augmented data set was able to identify transformed stop sign while the NN trained only on the original data was not. Moreover, the data was augmented in order to keep the same number of training examples with the same label, so I had around 2500 elements with one label.


#### 3. Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 Grayscale-Red-Blue image   				  |
| Convolution 5x5   | 1x1 stride, valid padding, outputs 28x28x6 	|
| ELU					      |												                      |
| Max pooling	      | 2x2 stride,  outputs 14x14x6  				      |
| Convolution 5x5	  | The same as above, outputs 10x10x16      		|
| Max pooling	      | The same as above, outputs 5x5x16       		|
| Fully connected		| Takes two pooled layers as the input, outputs onto 512 neurons        									|
| Dropout					  |	Using training is cutting half of the neurons											|
| Fully connected		| Outputs onto 256 neurons        					  |
| Dropout					  |	Using training is cutting half of the neurons											|
| Fully connected		| Outputs labels neurons        					  |
| Softmax				| Reports the percentage.        									|

Basically it is LeNet model discussed during the lectures with additional feed-forward link applied to the pooled layers to make the NN to understand wider concepts along the data.


#### 4. Training

To train the model, I used an Adam optimizer with rate of 0.001, batch size of 1024 and 100 epochs. Those parameters were chosen experimentally and have fit the model better than other values. However, the amount of epochs could be less, as the charts show that the NN was fitting the data at the same rate near 70 epoch.

#### 5. Results, research progress

My final model results were:
* validation set accuracy of 0.975
* test set accuracy of 0.964

An iterative approach was chosen:
* The first architecture I tried, was LeNet from lectures. It has shown an accuracy around 90% on the test set and after several tweaks and preprocessed data it has shown 91% on the test set.
* Later, I found a paper ["Traffic sign classification with
deep convolutional neural networks"](http://publications.lib.chalmers.se/records/fulltext/238914/238914.pdf) of Jacopo Credi, who used VGGNet. In my environment, it has shown result of 95% on the test set with augmentation and preprocessed images, moreover it was relatively heavy and hard to train because of huge amount of parameters.
* However, I was not fully satisfied with the results, so I tried the Conv2D architecture linked from the notebook, which was modifying LeNet to feed-forward convolutional layers. After training and parameter tweaking (increasing batch size and decreasing rate, adjusting augmentation angles and deviation), I received test set accuracy of 96.4%.

### New Images

#### 1. Additional German traffic signs

Here are ten German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11] ![alt text][image12]
![alt text][image13]

The third image might be difficult to classify because it is flipped a little bit. Moreover, second, fifth, sixth and eighth picture is a little bit different from ones in training set, as they were made in different country. Additionally, first image has number 3 similar to 5 and 7 from the test set.

#### 2. Model's predictions

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit of 30      		| Speed limit of 60   				|
| No passing for vehicles over 3.5 metric tons | Ahead only	|
| Stop					| Stop											|
| Dangerous curve to the right	      		| Dangerous curve to the right					 				|
| Double curve			| Dangerous curve to the right      							|
| Road work			| Road work      							|
| Traffic signals | Traffic signals |
| Children crossing | Speed limit (100km/h) |
| Ahead only | Ahead only |
| End of no passing | End of no passing |

The model was able to correctly guess 6 of the 10 traffic signs, which gives an accuracy of 60%. This compares to the accuracy on the test set of 97.5% and can be explained by slightly different signs given to the network.

#### 3. Softmax probabilities

The charts explaining the images top 5 softmax probabilities:

![alt text][image14]

For the first image, the model is relatively sure that this is a **speed limit of 60 km/h** (probability of **0.98**), whereas the image is containing sign of **speed limit of 30 km/h** with slightly different 3. Second top prediction with sign of **limit of 50 km/h** had probability of **0.02**, which was more similar to the original number.

| Prediction			        |     Probability	        					|
|:---------------------:|:---------------------------------------------:|
| 60 km/h | 97% |
| 30 km/h | 2% |
| Right-of-way at the next intersection | 0.15% |
| Slippery road | 0.10% |
| End of speed limit (80km/h) | 0.004% |

For the second image the network was sure(almost for **98%**) that it is a sign of **ahead only**, whilst it was **"No passing for vehicles over 3.5 metric tons"** sign. I feel like the model was confused with car contours and more white space than in training set. Table of predictions is:

| Prediction			        |     Probability	        					|
|:---------------------:|:---------------------------------------------:|
| Ahead only | 97.8% |
| Go straight or right | 1.1% |
| No passing for vehicles over 3.5 metric tons | 1.01% |
| Turn right ahead | 0.13% |
| Speed limit (120km/h) | 0.012% |

For the third and fourth images, as well as for the two last ones, the network did well with almost 99% right predictions.

The fifth image was got totally wrong with nothing right in first 5 places. The original sign was **double curve**, but the curve there was inverted, leading to mistakes in classification.

| Prediction			        |     Probability	        					|
|:---------------------:|:---------------------------------------------:|
| Dangerous curve to the right | 86.6% |
| General caution | 9.5% |
| End of all speed and passing limits | 4.4% |
| Right-of-way at the next intersection | 0.33% |
| Children crossing | 0.12% |

With the seventh image the prediction of **"Traffic signals"** was right, but it was almost mistaken with the **"General caution"** one, which can be described by their similarity.

| Prediction			        |     Probability	        					|
|:---------------------:|:---------------------------------------------:|
| Traffic signals | 50.2% |
| General caution | 46.5% |
| Bicycles crossing| 2.4% |
| Beware of ice/snow | 1.1% |
| Right-of-way at the next intersection | 0.11% |

With the eighth image, the network made a mistake, identifying **"Children crossing"** as **"Speed limit (100km/h)"**, due to similarity in general contours and line directions. However, right prediction was made on the first place too.

| Prediction			        |     Probability	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (100km/h) | 85.5% |
| Double curve | 7.4% |
| Children crossing | 5.8% |
| Road narrows on the right | 1.1% |
| Speed limit (120km/h) | 0.91% |

#### 4. Visualization of NN convolution layers

First layer of the NN with activations.
![alt text][image15]

This picture gives clear understanding, how the network "sees" the picture and interprets it. It detects some color changes, making several contrast/color height maps from it. Moreover, deeper layers detect more common shapes and their directions/positions, such as lines, rounds, as well as react to the special colors. These feature maps allowed me to understand, when the activations were triggered, therefore allowing to better preprocess the data.

Originally, my idea of splitting image into grayscale, yellow, red and blue went from those visualizations. However, the scheme with yellow has shown itself worse, so I decided to leave only the gray, red and blue ones.
