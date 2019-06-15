
# Stagnant zone segmentation with U-net
---

Originally u-net neural network architecture was built for performing semantic
segmentation on a small bio-medical data-set set [[Ronneberger et al.,
2015].](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

This deep neural networks is implemented with Keras.

## Overview
--------

Even though convolutional neural network (CNN) has recently become popular and
has increasingly been used as an alternative to many traditional pattern
recognition problems, it's application in material science is not common, for instance for segmenting stagnant zones of X-ray Computed
Tomography (CT) images.

In the present work, a U-net architecture is used for
segmenting the stagnant zone during silo discharging process. [You can find the full conference paper :
HERE](https://www.researchgate.net/publication/333787100_Stagnant_zone_segmentation_with_U-net).

### Architecture

<img src="https://github.com/selamgit/segmentation_with_U-net/blob/master/images/unet_architecture.png" width="600" height="600" title="Original scan and predicted 3D segmentation">

As each of the CT images already contain repetitive structures with the corresponding variation, only very few images are required to train a network that generalizes reasonably well. As a result, to make the u-net architecture work with very few training images, it has been modified to provide more accurate segmentation. 

Like on the original u-net, the modified architecture applied the same number of feature channels in upsampling part allow propagating context information to higher resolution layers.

One of the major modifications is that the original u-net used stochastic gradient descent optimizer (Ronneberger et al., 2015), but this modified u-net architecture used Adam optimizer (Kingma and Ba, 2014) to minimize the categorical cross-entropy objective.

### Data augmentation

Since the available dataset is small (only 30 images), an extensive amount of data augmentations
has been applied to improve the performance of the network.

The main goal of such augmentations is to prevent the network from memorizing
just the training examples and to force it to learn about the stagnant zone
boundaries. Therefore, in this study common transformation like rotation,
flipping, shifting, zooming, and shearing are applied.

During training, the transformations are applied on the fly so that the network
sees new random transformations during each epoch. Below table summarizes
applied augmentation methods and you could apply/modify the augmentation in
dataPrepare.ipynb.

| **Method** | **Range** |
|------------|-----------|
| Rotation   | \+/-20   |
| Flip       | 50%       |
| Shift      | 50%       |
| Zoom       | 50%       |
| Shear      | 50%       |

### Training

The dataset was first divided into two subsets, train, and test. The first
subset contains 30 images in which 80% were used for training and 20% for
validation. The trained model was next tested on the second subset which
contains the remaining 10 images.

During training, 30 manually annotated ground truth segmentations were used to
train the network to recognize the stagnant zone borders. The testing datasets
were used for the evaluation of the network performance.

The model was trained for 5 epochs.

After 5 epochs, calculated accuracy (Intersection over Union (IoU)) was about
**0.97** percent.

How to use
----------------------------------------

### Dependencies

This tutorial depends on the following libraries:

-   Tensorflow

-   Keras \>= 1.0

-   Python versions \>=2.7

### Run main.py

You could generate predicted results of test image in data/material/test

### Results

Use the trained model to generate predicted segmentation on test images. Some example images below: original and predicted segmentations:

<img src="https://github.com/selamgit/segmentation_with_U-net/blob/master/images/0_test.png" width="400" height="180" title="Original image and predicted segmentation">

<img src="https://github.com/selamgit/segmentation_with_U-net/blob/master/images/prediction.png" width="600" height="180" title="Original image, predicted segmentation and prediction mask">


### 3D segmentation
---------------

Once after having trained model using the CNN method, the end-to-end 3D
automatic segmentation offers an effective and fast segmentation of stagnant
zone.

In order to prove that the trained model could generate the stagnant zone
segmentation for completely different scan (the model was trained on sorghum
grains flow scan), it was tested by using a 3D scan of rice grains flow. The
result shows that the trained model was able to generate predicted segmentation
successfully.

<img src="https://github.com/selamgit/segmentation_with_U-net/blob/master/images/3D_unet_segmentation.png" width="400" height="180" title="Original scan and predicted 3D segmentation">

### Citation
If you find our work useful, please cite:
```
@inproceedings{2019 IEEE Second International Conference on Artificial Intelligence and Knowledge Engineering (AIKE),
    title = {Stagnant zone segmentation with U-net}
    author = {Selam Waktola, Krzysztof Grudzien, Laurent Babout},              
    Conference = {Artificial Intelligence & Knowledge EngineeringAt: Cagliari, Italy},
    year = {June 2019}
}
```

### REFERENCE
[Ronneberger, O., Fischer, P., Brox, T., 2015 : U-Net: Convolutional Networks for Biomedical Image Segmentation, in: Medical Image Computing and Computer-Assisted Intervention (MICCAI). Springer, pp. 234–241.](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

[Kingma, D.P., Ba, J. 2014 : Adam: A Method for Stochastic Optimization. Int. Conf. Learn. Represent.](https://arxiv.org/pdf/1412.6980.pdf)

[Waktola, S., Grudzień, K., Babout, L. 2019 : Stagnant zone segmentation with U-net: 2019 IEEE Second International Conference on Artificial Intelligence and Knowledge Engineering (AIKE), Cagliari, Sardinia (Italy).](https://www.researchgate.net/publication/333755407_Stagnant_zone_segmentation_with_U-net)

