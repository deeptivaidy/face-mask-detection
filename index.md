## CS 4641: Face Mask Detection
Project can be found at [deeptivaidy.github.io/face-mask-detection](https://deeptivaidy.github.io/face-mask-detection)

Contributors (alphabetically): Tushna Eduljee, Kenneth Kanampully, Nesha Prabahar, Deepti Vaidyanathan


### Introduction and Background

COVID-19 has truly changed the world and has forced everyone to rethink how we operate our day to day lives. From working from home to social distancing, life in 2020 is very different from life in 2019. One of these major changes is the introduction of masks as a public health measure to reduce the spread of this disease. However, a lot of citizens have pushed against mask mandates and will not comply with policies of business by not wearing their mask. This causes a public health risk and a hassle for business owners who have to ensure that all of the customers are wearing a mask. To combat this issue, it would be very helpful to have an automated way to detect the people who do not wear masks in these enclosed spaces.

In the past, there have been similar projects that have been aimed at classifying the different types of masks (i.e. cloth mask, surgical mask, or no mask). Instead of just classifying these images, we want to take this project to the next step and perform an object detection task, where we aim to draw a bounding box around all people in a given image who are not wearing a mask.

### Methods
We aim to use unsupervised learning for data processing and supervised learning for the remaining model. Since our project deals with high volumes of image data, and our model will require quite a bit of processing power in order to train, we plan on using some form of unsupervised learning in order to reduce the image dimensionality or feature space as it comes into our model. In order to do so, we aim to look into PCA (principal component analysis) in order to reduce the dimensionality of incoming images.

In addition, we will use supervised learning in order to idenify the people in the frame who are not wearing a mask using existing object detection architectures. To train our model, we found a [kaggle dataset](https://www.kaggle.com/sigmind/masked-face-detection-wider-dataset?select=COVID-mask-detection_WIDER.tar.xz) that has labelled bounding boxes around mask wearers and non-mask wearers. After studying various object detection models (YOLO, SSD, Faster-RCNN, and R-FCN), we have decided to use the Faster-RCNN model because it has very high accuracy without sacrificing speed. Compared to the other models, which make various trade-offs in accuracy and confidence to work well with multiple frames of data, Faster-RCNN is the most accurate for our purposes since we are only focusing on single-frame image data and do not need to be concerned about multi-frame detection at this time.

### Results


### Discussion
If we are able to achieve our goals, we would have built a way to accurately detect multiple people who are not wearing a mask in a picture. This could then be implemented in security cameras in enclosed spaces, allowing business owners to do real time monitoring of the people in their stores.

Another potential use of this technology would be to use it to detect the most common areas indoors or outdoors where people do not have masks. One could then take this information and use it to avoid the high risk areas where people do not comply with mask mandates.

### References

![Project Overview Infographic](infographic.png)

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/deeptivaidy/face-mask-detection/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
