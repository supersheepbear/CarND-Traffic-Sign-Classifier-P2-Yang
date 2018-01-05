# **Project: Build a Traffic Sign Recognition Program** 
## Yang Xiong

### This project is the second project of self driving nano degree

Overview
In this project, I used deep neural networks and convolutional neural networks to classify traffic signs. I trained and validated a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). 

### Dataset Summary
Below is a basic summary of the data set:

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
training set: validation set: testing set = 0.67 : 0.09 : 0.24

Here are some visulizations of the dataset:
#### train,validation,test set distribution
We have 67% of training set, 9% of validation set and 24% of test set.
As the dataset is large, we don't need too much validation set. Otherwise, more validation set shall be needed to prevent overfitting.
![](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/report_images/data_ratio.png)
#### unique label data distribution
Here I plot the data number versus unique label for each dataset. 
This is important because we want the train,validation and test sets to have data from all different labels.<br>
Otherwise, if data from some labels are missing from one of the data set, it will  either affect the training performance or the test performance.
![](https://github.com/supersheepbear/CarND-Traffic-Sign-Classifier-P2-Yang/report_images/unique_label_distribution.png)
As shown above, our data set have similar distributions for unique labels, which is good.


