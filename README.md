# **Project: Build a Traffic Sign Recognition Program** 
## Yang Xiong

## This project is the second project of self driving nano degree

Overview
In this project, I used deep neural networks and convolutional neural networks to classify traffic signs. I trained and validated a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 

# 1. Dataset Summary
Below is a basic summary of the data set:
```
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
training set: validation set: testing set = 0.67 : 0.09 : 0.24
```
Here are some visulizations of the dataset:
## 1.1 train,validation,test set distribution
We have 67% of training set, 9% of validation set and 24% of test set, as summarized below.
As the dataset is large, we don't need too much validation set. Otherwise, more validation set shall be needed to prevent overfitting.
```
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
training set: validation set: testing set = 0.67 : 0.09 : 0.24
```

![data_ratio](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/data_ratio.jpg)
## 1.2 unique label data distribution
Here I plot the data number versus unique label for each dataset. 
This is important because we want the train, validation and test sets to have similar distribution from all different unique labels.<br>
Otherwise, if data distrubutions are very different for these data set, it will  either affect the training performance or the test performance.
![unique_label_distribution](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/unique_label_distribution.jpg)
As shown above, our data set have similar distributions for unique labels, which is good.

## 1.3 examples of stop signs data
Below are random examples from each unique label stop sign data.
There are 43 different types of stop signs.
![stopsign_examples](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/stopsign_examples.jpg)

## 1.4 Image augmentation 1st attempt(heavy augmentation)
**This is the first attemp of my image augmentation(not used for final model)**<br>
Here I use image augmentation library, and define a function to randomly augment images using augnmentation techniques
The augmentation techniques include:
- Flip image left and right
- crop the image 
- apply Gaussian Blur to images
- Strengthen or weaken the contrast in each image
- Add gaussian noise
- make images brighter or darker
- Apply affine transformations to each image(Scale/zoom them, translate/move them, rotate them and shear them)
```python
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.ContrastNormalization((0.75, 1.5)),
    # Add gaussian noise.
    # For 50% of all images, we sample the noise once per pixel.
    # For the other 50% of all images, we sample the noise per pixel AND
    # channel. This can change the color (not only brightness) of the
    # pixels.
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order
```
In the 1st attemp, I use above function to randomly augment images<br>
It applies crops and affine transformations to images, flips some of the images horizontally, adds a bit of noise and blur and also changes the contrast as well as brightness.<br>
Through this process we now have 10 times more images for our traning set, which is a huge addition
>**Note: this data set is not used for final model, because after testing, I found that I have added too much noise to the training set, which leads to degregation of of model performance**

examples of 1st attemp augmentation images:
![augmentation_1st_samples](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/augmentation_1st_samples.jpg)

## 1.5 Image augmentation 2nd attempt(light augmentation)
**This is the second attemp of my image augmentation(This augmentation data set is used for final model)**<br>
From the 1st attemp I found that I should not add too much noise on images. The reason is that the augmented images may be impossible for machine or human beings to interpret after heavy agumentation.
So in the 2nd try of augmentation, I use less heavy augmentation techniques to get a reasonable traning set. This ends up getting better perfomance in my final model.
Below is the list of augmentations.

- make images brighter or darker
- Apply affine transformations to each image(Scale/zoom them, translate/move them, rotate them and shear them)

Now process original training images and save to a local file train_aug_adjust.p<br>
The new training set is 5 times larger than the original training set.
Here are examples of 2st attemp augmentation images. We can see that it is not heavy augmentation like the 1st attempt

![augmentation_2nd_samples](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/augmentation_2nd_samples.jpg)

Through augmentation, now we have:
```
Number of augmented training examples = 208794
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
aug training set: validation set: testing set = 0.92 : 0.02 : 0.06
```
Now the data set ratio is as below. We have much more training set than the original one. The new training set is store as train_aug.p<br>
![augmentation_2nd_samples](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/data_ratio_augment.jpg)

# 2. Design and Test a Model Architecture
The preprocessing includes:
- apply normalization for images. Normalized inputs are easier for model parameters to get small and reasonable mean and variance
- apply grayscale to images. This is in order to give less input for our model to train on. 
Here are examples of normalized and grayscale images:

![stopsign_gray_and_norm.jpg](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/stopsign_gray_and_norm.jpg)

final input image shape fed to Convnet shape is : (208794, 32, 32, 1).
Note that data are shuffled before fed into Convnet

# 3 Model development progress
## 3.1 model defining
**Here is the network of my training layers.<br>**
Basically what I use is the original LeNet achitecture, plus an incetion layer in the middle, which provides better performance and better converging speed.<br>
![lenet.png](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/lenet.png)

For the inception layer, here is the achitecture:
![inception_layer.png](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/inception_layer.png)


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 normalized grayscale image   			| 
| Convolution#1	     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU#1				|												|
| Max pooling#1	      	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Convolution#2  		| 1x1 stride, valid padding, outputs 10x10x16   |
| RELU#2				|           									|
| Max pooling#2	      	| 2x2 stride, valid padding, outputs 5x5x16  	|
| inception	      	| 1x1,3x3,5x5, averagepool(same pad), outputs 5x5x31  	|						|    									|
| Flatten#2				| outputs: 400    								|
| Full Connection#0		| outputs:120	       |
| Full Connection#1		| output: 84									|
| Full Connection#2		| output: 43									|
| Output log    			|     		43 logits           |

## 3.2 Model development
### 3.2.1 Baseline training
I start with a baseline training with final training accuracy 0.99 and final validation accuracy 0.92.<br>
For this baseline training I use:
- batchsize : 64
- learningrate: 0.001
- optimizer: Adam
- EPOCH: 20
- layers: original Lenet
- regularizations: no<br>
<br>

**problem 1: It seems like the batch size too small affect the stability**<br>
**problem 2: learning rate and other hyper parameters can be tuned to be better***<br>
**problem 3: This is clearly overfitting!**<br>
So What I do is to do a lot of experiments on regularization techniques, and hyperparameters tuning<br>


### 3.2.2 Dropout keepprob experiment

For this experiment , I add dropout technique to the two FC layers of Lenet, and try different keepprob value from 0.3 to 1.0, to see the accuracy and loss performance.<br>

I Save the parameters I use into a csv file, and save plots into a pdf file in 'hyper_para_testing\dropout\dropout_test.pdf'<br>
Through all the test result, I found out that dropprob 0.5 has the best performanc, as shown below.<br>

Through dropout keepprob 0.5 I increase the validation accuracy performance from 0.94 to 0.96
![dropout_sample.PNG](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/dropout_sample.PNG)

### 3.2.3 L2 Regularization test

For this experiment , I combine L2 norm with dropout technique to see what happends to the performance.<br>
I multiply 0.001 to the L2 term and add it to the loss function
I Save the parameters I use into a csv file, and save plots into a pdf file in 'hyper_para_testing\dropout_and_L2_norm\dropout_and_L2norm.pdf'<br><br>
**Through the test result, I found out that L2 norm does not increase my performance with already having dropout, as shown below.**<br>
Therefore I choose not to use L2 norm.
![L2_norm_sample.PNG](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/L2_norm_sample.PNG)

### 3.2.4 batch norm experiment
 I've heard that batch norm can increase model performance, and having slight regularization effect on models.<br>
 Here I add batch norm layers and try it with/without dropout to see what heppens.
 You can find the whole results from :'hyper_para_testing\batchnorm\batchnorm.pdf'
 
 Below is the result of batch norm combined with dropout. **It seems that it hurts the performance somehow. I assume normally it won't hurt model performance, here it maybe because I already normalize the input, so it does not make much contribution to my model.**
<br>
 In the later experiment, I did not use batch norm.
 ![batchnorm_sample.PNG)](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/batchnorm_sample.PNG)
 
### 3.2.5 rate decay/SGD with momentum experiment
SGD normally has better performance in small data set according to some essays.<br>
Also, rate decay normally contributes to model converging.
Here what I try is using momentum optimizer with rate decay, instead of Adam, to see what happens.

 You can find the result from :'hyper_para_testing\rate_decay\rate_decay.pdf'
 I use:
 -initial weight: 0.025
 -ratedecay: 0.85(For ecah epoch, rate = rate * ratedecay)
 -momentum factor:0.9

 Below is the plot with the result.
 **It seems for my model, SGD with momentum and rate decay is much more stable than Adam, so I decide to use momentum optimizer for my later model**
<br>
 ![rate_decay_sample.PNG](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/rate_decay_sample.PNG)

### 3.2.6 experiment with constrained softmax loss layer/Monte carlo simulation

These are more like experiments of my personal interetes. They do not contribute to my final model, So I don't show them here. <br>If you are interested to see these results, please look into the hyper_para_testing folder for the results.<br>
Basically, constrained softmax loss layer is from someone's paper to say that it has regularization effect.<br>
Monte carlo simulation is to try randomly picking hyperparameters to get possible nice parameters.
###  3.2.7 inception layer experiment
Inception layer is a Google developed layer which helps model performance and converging.<br>
Here I use some hyperparameters from monte carlo simulation and try inception layer on Lenet. <br>
We can see that the for my current data set and model it does not do anything. I think for more complex model or data set, it will work.
 ![inception_sample.PNG](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/inception_sample.PNG)
### 3.2.8 Data augmentation experiment
From above I have achieved 96% accuracy for validation set. but it's still overfitting!<br>
So I decide to try the data augmentation technique.<br>
As stated in section 2, I have try two different augmentation data set.<br>
The 1st augmentation training set is a heavy augmentation data set, which is very noisy.<br>
I end up not able to get better training accuracy or validation accuracy more than 94%.<br>
As a result, I use the 2nd augmentation training set. My final model with this augmentation training set gets very good result as shown in next section 3.3

## 3.3 Final model
**For final model I have achived 98.3% accuracy for training set, and 98.0% for validation set**<br>
Note that for final model I used the inception layer, because it significantly increases my training speed, and does not seem to hurt my performance.<br>
For final model I use, below are the details and hyperparameters used for the model:

### Hyper parameters
- dropout keepprob : 0.5
>*The dropout keepprob is tuned to be 0.5 to get reasonably good regularization on data*
- initial rate: 0.035
- rate decay: 0.88 (rate  = 0.88 * rate for each EPOCH)

>*I use rate, rate decay and some logic for rate.<br>
example for rate logic:
```python
        if train_accuracy<0.97 and rate<0.01:
            n_rate = 0.01
        elif train_accuracy>0.97 and train_accuracy<0.98 and rate<0.004:
            n_rate = 0.005   
        elif train_accuracy>0.98 and train_accuracy<0.983 and rate<0.0015:
            n_rate = 0.004
        #elif train_accuracy>0.98 and train_accuracy<0.983 and rate<0.0015:
            n_rate = 0.002
        elif train_accuracy>0.983 and train_accuracy<0.985 and rate<0.008:
            n_rate = 0.0015
        elif train_accuracy>0.985 and train_accuracy<0.987 and rate<0.0004:
            n_rate = 0.0008
```
These rate conditions are set based on fine tuning them on the augmentation training set, to successfully converge to a good optimal point. The augmentation training set is harder than the original training set to be trained on. When I tried fixing rate or just simple rate decay, the model seems to easily stuck on local optimal point. Through these conditions I am getting very good accuracy.*
- batch size: 512 for the 1st part ot training, and 4096 for the 2nd part ot training.
>*I choose this batch size because my 1080ti is able to handle this amount of memory, and the model converges fast enough and to a reasonable point.*
- EPOCHS: 210 EPOCHS
>*The training accuracy seems hard to grow after this EPOCHS. I think is a reasonable eraly stop point to prevent overfitting*
- L2 norm: not used
>*I got good regularization from dropout so I don't bother L2 norm*
- momentum factor:0.9
>*0.9 seems good on training process*


### Final model training 1st part. 
In this first part of training, I use the hyper parameters as shown above, and let the model stops at training accuracy 98% and validation accuracy > 98%. This end out using 112 EPOCHS. 

### Final model training 2nd part. 
Since I doubt that the model performance can be improved futherly, I decided to train more epochs to get better final performance.<br>
What I do is that I load the model just trained on, and train some more EPOCHS, and let the model stop at training accuracy> 98.4% and validation accuracy > 98.3%. This end out using another 98 EPOCHs for the training.<br>
I this period, I set tighter learning rate and batch size 4096 for a better converging effect. <br>
The final model performance is shown in the following figure.
The final model is saved to '.\lenet_final'

 ![final_model.jpg](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/final_model.jpg)
Here are some of my answers for the questions:<br>
What architecture was chosen?<br>
Lenet 5 with an incetion layer. 
Why did you believe it would be relevant to the traffic sign application?
Because it has good effect on the classification for numbers, I believe it shall has similar performance for stop signs, since they are both multiclass classification problem.<br>
How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The training, validation of more than 98% indicates that the model works well on th training part, and it's not overfitting/underfitting. The 95.7% one test set shows that it works well on out of sample prediction. However, further improvement can be included to get it better. 
## 3.4  evaluation of the test set
Our model accuracy on the test set is 95.7%.
Futher improvement can be made by:
- adding more diversified validation set, training set
- improve model achitecture
- better tuning hyperparameters
- better training process<br>
**Below are the precision and recall for the model, which may give us infomation about which label performance needs to improve for my model.**
```


label0  recall:0.8833 precision:0.8983
label1  recall:0.9875 precision:0.9556
label2  recall:0.9733 precision:0.9505
label3  recall:0.9044 precision:0.9532
label4  recall:0.9652 precision:0.9830
label5  recall:0.9413 precision:0.9095
label6  recall:0.9400 precision:0.9463
label7  recall:0.9222 precision:0.9857
label8  recall:0.9911 precision:0.9272
label9  recall:1.0000 precision:0.9677
label10  recall:0.9894 precision:0.9969
label11  recall:0.9357 precision:0.9269
label12  recall:0.9870 precision:0.9913
label13  recall:0.9944 precision:0.9958
label14  recall:1.0000 precision:0.8882
label15  recall:0.9905 precision:0.9952
label16  recall:1.0000 precision:1.0000
label17  recall:0.9361 precision:0.9825
label18  recall:0.8641 precision:0.9656
label19  recall:0.9667 precision:0.9831
label20  recall:1.0000 precision:0.9574
label21  recall:0.7222 precision:0.8904
label22  recall:0.9000 precision:1.0000
label23  recall:0.9933 precision:0.8514
label24  recall:0.8778 precision:0.8681
label25  recall:0.9875 precision:0.9151
label26  recall:0.9778 precision:0.9462
label27  recall:0.5667 precision:0.7556
label28  recall:0.9800 precision:0.9735
label29  recall:0.9111 precision:0.8723
label30  recall:0.7400 precision:0.7551
label31  recall:0.9926 precision:0.9926
label32  recall:1.0000 precision:0.9836
label33  recall:0.9810 precision:0.9810
label34  recall:1.0000 precision:0.9836
label35  recall:0.9590 precision:0.9973
label36  recall:0.9750 precision:0.9590
label37  recall:0.9500 precision:0.9500
label38  recall:0.9536 precision:0.9821
label39  recall:0.9556 precision:0.9053
label40  recall:0.8889 precision:0.8989
label41  recall:0.8000 precision:0.9796
label42  recall:0.8889 precision:0.9877
```

# 4. Test a Model on New Images

I have download at 8 pictures of German traffic signs from the web and use my model to predict the traffic sign type.<br>
Below are the images after resize.
![web_test_images.png](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/web_test_images.png)

Then I predict  the sign type for each image:<br>
```
label value for test1.jpg: 1, predict value for test1.jpg: 1

label value for test2.jpg: 27, predict value for test2.jpg: 27

label value for test3.jpg: 40, predict value for test3.jpg: 40

label value for test4.jpg: 11, predict value for test4.jpg: 11

label value for test5.jpg: 25, predict value for test5.jpg: 25

label value for test6.jpg: 38, predict value for test6.jpg: 38

label value for test7.jpg: 17, predict value for test7.jpg: 17

label value for test8.jpg: 18, predict value for test8.jpg: 18
```

As you can see, for 8 web images, our model accuracy is 100%.Cool! Awesome!<br>
the images I choose is not hard for model (and for human beings) to classify.  I guess it will be hard if I pick some images that are harder to classify.<br>

Here I output top 5 softmax probabilities for each image found on the web:
```
top 5 Softmax Probabilities for test1.jpg:
probabilities:95.008 | 70.116 | 68.233 | 58.279 | 40.288

predict label:  1    |   0    |   2    |   4    |   3    

Top probability is label 1, which is predicted as:Speed limit (30km/h)
--------------------------------------------------------------------------------
top 5 Softmax Probabilities for test2.jpg:
probabilities:76.475 | 62.935 | 56.542 | 54.258 | 52.051

predict label: 27    |  11    |  30    |  25    |  21    

Top probability is label 27, which is predicted as:Pedestrians
--------------------------------------------------------------------------------
top 5 Softmax Probabilities for test3.jpg:
probabilities:39.476 | 37.008 | 35.916 | 26.224 | 22.053

predict label: 40    |   7    |  12    |   1    |  39    

Top probability is label 40, which is predicted as:Roundabout mandatory
--------------------------------------------------------------------------------
top 5 Softmax Probabilities for test4.jpg:
probabilities:117.771 | 97.103 | 66.227 | 51.615 | 49.600

predict label: 11    |  30    |  27    |  37    |  21    

Top probability is label 11, which is predicted as:Right-of-way at the next intersection
--------------------------------------------------------------------------------
top 5 Softmax Probabilities for test5.jpg:
probabilities:225.522 | 113.338 | 108.304 | 107.948 | 105.992

predict label: 25    |  20    |  28    |  18    |  22    

Top probability is label 25, which is predicted as:Road work
--------------------------------------------------------------------------------
top 5 Softmax Probabilities for test6.jpg:
probabilities:235.531 | 115.659 | 107.046 | 106.958 | 99.750

predict label: 38    |  34    |  12    |  13    |  10    

Top probability is label 38, which is predicted as:Keep right
--------------------------------------------------------------------------------
top 5 Softmax Probabilities for test7.jpg:
probabilities:110.727 | 77.177 | 68.362 | 66.750 | 51.072

predict label: 17    |  38    |  12    |  39    |  34    

Top probability is label 17, which is predicted as:No entry
--------------------------------------------------------------------------------
top 5 Softmax Probabilities for test8.jpg:
probabilities:43.625 | 39.750 | 32.196 | 32.076 | 28.563

predict label: 18    |  26    |  27    |  24    |  25    

Top probability is label 18, which is predicted as:General caution
--------------------------------------------------------------------------------
```
It seems that model is very certain about image number 1,2,4,5,6,7, but is not very certain about image 3 and 8.<br>
Let's visualize what it is confused with for test3.jpg:<br>

 ![confused_image_1.jpg](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/confused_image_1.jpg)
 
Well, I think I don't understand why model is confused about these two images... But the model has its reason. <br>
For test8.jpg, here is the visulization of the confused image:
 ![confused_image_2.jpg](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/confused_image_2.jpg)
 
For test8.jpg, this is understandable. The confused labe image looks really similar to the true lable image. It's hard even for human beings.<br>
### Compare the performance on the new images to the accuracy results of the test set.
For the test set we have 95.7% accuracy, while for new images we have 100% accuracy.<br>
This comparison is actually not fare because we have too little amount of images for the new images.<br>
I think if I increase the number of new images, the comparison will make more sense.


# 4. Visualize the Neural Network's State with Test Images:
Take an example of the following input image:
 ![Visualize_raw.png](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/Visualize_raw.png)
 
 Below are some visulizations of the layers output:<br>
 conv1:
  ![conv1.png](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/conv1.png)
 conv1_act:
  ![conv1_act.png](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/conv1_act.png)
 conv1_pool:
  ![conv1_pool_act.png](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/conv1_pool_act.png)
 conv2:
  ![conv2.png](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/conv2.png)
 conv2_act:
  ![conv2_act.png](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/conv2_act.png)
 inception:
  ![inception_visual.png](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/raw/master/report_images/inception_visual.png)
