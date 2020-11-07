# CS 4641: Face Mask Detection

Contributors: Tushna Eduljee, Kenneth Kanampully, Nesha Prabahar, Deepti Vaidyanathan


## Project Goal

COVID-19 has truly changed the world and has forced everyone to rethink how we operate our day to day lives. From working from home to social distancing, life in 2020 is very different from life in 2019. One of these major changes is the introduction of masks as a public health measure to reduce the spread of this disease. However, a lot of citizens have pushed against mask mandates and will not comply with policies of business by not wearing their mask. This causes a public health risk and a hassle for business owners who have to ensure that all of the customers are wearing a mask. To combat this issue, it would be very helpful to have an automated way to detect the people who do not wear masks in these enclosed spaces.

## Project Overview

In order to gain a better idea of this area of research and development, and ensure that our project would be building upon what was already in the field, we decided to conduct some background research. We found projects that have worked to classify different types of masks on an individual (one person in the frame and they are wearing a mask), such as cloth and surgical. Although this project also has useful applications in prevention and preventative care, we decided that we would want to cater our project towards an object detection task in order to change the target audience. While prior projects often feature the user of masks themselves as the user of the model, our model has applications in the enterprise world. We aim to perform an object detection task in our project, such that our model draws bounding boxes around those in an image who are not wearing a mask.

In the development of our model, we have decided to use unsupervised learning for the data processing pipeline and supervised learning for the model itself. Since our project deals with high volumes of image data, our model will require quite a bit of processing power in order to train. For this reason, we have used principal component analysis (PCA) in order to reduce the image dimensionality and feature space, before the image is fed into our ML model. This allows the model to have a much more condensed and less sparse image, thereby training less feature weights, and enabling our model to not only run faster but also be less likely to overfit as our dataset is on the smaller side for an object detection task.

For the model selection task, we conducted thorough research on YOLO, Faster RCNN, RFCN, and SSD, in order to gauge which would be more fitting for our particular object detection task. One particular Google research paper informed much of our decisions on the different use cases and specs of these architectures. We will likely be choosing Faster RCNN for our project, as it maintains very high accuracy with some sacrifices for speed. For our purposes, this works well as our model is going to be running locally rather than on embedded device and we don’t currently have plans of running an MVP in real time.

![Project Overview Infographic](infographic.png)

## Touchpoint 2 Goals

Our goals for Touchpoint 2 are:
1. Clean the data
1. Process the data using principal component analysis (PCA)

## Dataset

The dataset that we found would be most suitable for our needs can be found on kaggle as the Face Mask Detection dataset. One key aim of our model is that it can identify many people in a crowd and subsequently whether or not they are wearing a mask. For this reason, this dataset of people in crowds is perfect for our needs. As our model is a supervised learning model, it was also important that the dataset be pre-labeled with important needed information. The labels that come with this dataset include labels for people wearing masks, those not wearing them, and those them improperly. As each image features more than one person, it cannot have a raw label, as there may be a mix of people wearing their masks in different ways per image. For this reason, this dataset also contains the bounding box labels for the people in the images along with the appropriate labels for each box (one of three classes of labels). The bounding boxes are in PASCAL VOC format.

[Here is the link to the kaggle dataset](https://www.kaggle.com/andrewmvd/face-mask-detection)

Example Images:

![Example Image 1](dataset1.png)
![Example Image 2](dataset2.png)

## Data Cleaning Procedures

#### Cleaning Bounding Box XML Files

The first step in our data cleaning efforts includes converting the xml files (one per image) into tabular data frame format so that it could be used for preprocessing purposes. This consisted of having an individual column for each possible bounding box in an image. Although this leads to a somewhat sparse array, we manage this by reducing outliers as well as using the numpy special representation for sparse arrays.

#### Removing a Classification from the Dataset

For our model, we really only require class one (wearing a mask) or class two (not wearing a mask). For this reason, we are removing the bounding box labels which belong to class three (wearing a mask, but improperly). This is because those images vary widely from wearing a mask slightly improperly to barely wearing a mask. As there was too much ambiguity, we decided to remove that category altogether and focus our model on wearing or not wearing a mask. This results in a case of skewed classes, but we prefer the case of people being labelled with a mask to without, in case our model is deployed in the real world. This is because resources of locating the person in a business establishment should be saved for those truly not wearing a mask and not those half-wearing a mask. 81% of our labelled bounding boxes are with masks, and all other labels are without masks - showing the skewed classes.

Example Image of incorrectly wearing a mask:

![Example Image 3](incorrect.png)

#### Removing Outliers

The final stage in our data cleaning procedure was removing extreme outliers from the dataset so that we can reduce our feature space from 120 columns to 40 columns. An outlier for our purposes is an image with an abnormally large amount of bounding boxes associated with it. This would be an outlier because from a data processing standpoint, there would be no need to increase the dimensionality of the data being processed by ten fold just for the sake of a few images from the entire bunch having many faces detected. The identification of such outliers was done based on the number of bounding boxes per image and the box plot of that data. We do not get rid of all outliers, however, as there is a significant enough portion to include in our dataset so that we can show an accurate representation of different sized crowds. Additionally, this removal of outliers further decreases the dataset which is already on the smaller side. This cleaning reduces it by 7 images.

> Graphs on outliers

## Data Preprocessing

For the data preprocessing stage we first decided to standardize image sizes by adding padding to the bottom and left edges of each image so that it increased the overall image size to 600x600. This effectively became the maximum x and y for all images in dataset. Although increasing the representation of each image effectively means storing more data, and therefore performing more computation, it also means that we do not have to crop images and therefore edit any bounding box information. Once we complete our model, we plan on looking into further performance improvements such as scaling the image in order to reduce the need for padding.

Additionally, in order to prep the data for the pipeline, we have decided to use principal component analysis (PCA) before the images reach the supervised learning model itself. PCA is a form of unsupervised learning which uses variance and covariance of a dataset in order to take linear combinations of the data and reduce the overall dimensionality without resulting in much data loss (depending on hyperparameter tuning of k based on percentage of variance retained). It is used to speed up the running time of an algorithm. We decided to use PCA for our data pipeline in order to reduce image representation in memory and therefore require less computation per image passing through our model.

> Images of before and after PCA

## Future Work

Training Model: In order to train our model, we have found a kaggle dataset that has labelled bounding boxes around mask wearers and non-mask wearers. The kaggle dataset can be found at this link: https://www.kaggle.com/notadithyabhat/face-mask-detector

Evaluation Method: We are planning to evaluate our methods using Intersection over Union (IoU) and mean average precision (mAP). IoU determines overlap of bounding boxes with the group truth to create a threshold for mAP and accordingly adjust the weights. mAP is the area under the precision-recall curve. A prediction is correct if it meets the threshold IoU. 

As we have just completed our unsupervised learning step, and PCA did not require us to convert categorical data into discrete, we have yet to make the design choice of whether we want to encode the categorical data as one hot or integer encoding.

Additionally, once we complete our model, we plan on looking into further performance improvements such as scaling the image in order to reduce the need for padding.


## References
1. “Face Mask Detection System Using AI: AI Mask Detection.” Software Development Company, www.leewayhertz.com/face-mask-detection-system/. 
2. “Faster R-CNN: ML.” GeeksforGeeks, 1 Mar. 2020, www.geeksforgeeks.org/faster-r-cnn-ml/. 
3. Gandhi, Rohith. “R-CNN, Fast R-CNN, Faster R-CNN, YOLO - Object Detection Algorithms.” Medium, Towards Data Science, 9 July 2018, towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e. 
4. Jaadi, Zakaria. “A Step by Step Explanation of Principal Component Analysis.” Built In, builtin.com/data-science/step-step-explanation-principal-component-analysis. 
5. Katz, Josh, et al. “A Detailed Map of Who Is Wearing Masks in the U.S.” The New York Times, The New York Times, 17 July 2020, www.nytimes.com/interactive/2020/07/17/upshot/coronavirus-face-mask-map.html. 
6. Koech, Kiprono Elijah. “On Object Detection Metrics With Worked Example.” Medium, Towards Data Science, 10 Sept. 2020, towardsdatascience.com/on-object-detection-metrics-with-worked-example-216f173ed31e. 
7. Ren, Shaoqing, et al. “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.” ArXiv.org, 6 Jan. 2016, arxiv.org/abs/1506.01497. 
8. Rosebrock, Adrian. “Intersection over Union (IoU) for Object Detection.” PyImageSearch, 18 Apr. 2020, www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/. 
9. Sigmind. “Masked Face Detection In the Wild Dataset.” Kaggle, 19 June 2020, www.kaggle.com/sigmind/masked-face-detection-wider-dataset?select=COVID-mask-detection_WIDER.tar.xz. 
