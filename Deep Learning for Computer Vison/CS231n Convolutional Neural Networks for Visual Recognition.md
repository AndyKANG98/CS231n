# Convolutional Neural Networks for Visual Recognition
 This course is a deep dive into details of the deep learning architectures with a focus on learning end-to-end models for these tasks, particularly image classification. During the 10-week course, students will learn to implement, train and debug their own neural networks and gain a detailed understanding of cutting-edge research in computer vision. The final assignment will involve training a multi-million parameter convolutional neural network and applying it on the largest image classification dataset (ImageNet). We will focus on teaching how to set up the problem of image recognition, the learning algorithms (e.g. backpropagation), practical engineering tricks for training and fine-tuning the networks and guide the students through hands-on assignments and a final course project. 
 --- 
 <br>
 <br>
 
 ## Introduction
 > * Image Classification
 > * Object detection
 > * Image Captioning
 
Convolutional Neural Networks have become an imortant tool for object recognition <br>

* Overview
  * Thorough and Detailed.
    * Understand how to write from scratch, debug and train convolutional neural networks.
  * Practical.
    * Focus on practical techniques for training these networks at scale, and on GPUs (e.g. will touch on distributed optimization, differences between CPU vs. GPU, etc.) Also look at state of the art software tools such as TensorFlow, and PyTorch
<br>

## Image Classification
 * Challenges: 
   * Viewpoint variation 
   * Illumination (light) 
   * Deformation (shape)
   * Occlusion (part)
   * Background clutter
   * Intraclass variation (different appeareance)
  
 * Data-driven approach:
   * Collect the dataset of images and labels
   * Use Machine Learning to train a classifier
   * Evaluate the classifier on new images
  
 * Data Matric
 
 * K-Nearest Neighbors
 
 * Setting Hyperparameters
   * #4 Cross-validation
  
 * Linear Classification
   * f(x,W) = Wx+b     W is the weight matrix (col: numn of pixel; row: num of classification)
   * only learn one template of the object

<br>

## CNN for Visual Recognition

* Loss function

* regularization

* Softmax Classifier (Multinomial Logistic Regression)

* Optimization
  * set the step size and learning rate

## Neural Network for Visual Recognition
* backpropagation (the computational graph - calculate the gradiant)

## Convolutional Neural Networks for Visual Recognition
very useful in classification; object detection (bounding box); Image caption; Art work

1. Convolutional Layer
filters are feature identidiers to describe the image
Output size: (N - F)/stride + 1

2. Pooling laeyr 
 makes the representation smaller and managebal
 operates over each activation map individually
3. Fully Connected layer
out put a N dimentional output

## Train Nerual Networks
* Activation Function (reLu)
* Weight Initialization
* Preprocess the data
* Babysitting Learning
* Hyper parameter Search
* Optimization (SGD, Momentum, AdaGrad, RMSProp)
* Regularzation (Data augmentation)
* Transfer Learning

## Hardware and Software
* CPU, GPU, TPU
* PyTorch (version 0.4) - Like a numpy array, but can tun on GPU; easy to develop and debug 
* Tensorflow - High-Level Wrappers

## Cases
Case Studies
- AlexNet
- VGG
- GoogLeNet
- ResNet
Also....
- NiN (Network in Network)
- Wide ResNet
- ResNeXT
- Stochastic Depth
- Squeeze-and-Excitation Network
- DenseNet
- FractalNet
- SqueezeNet
- NASNet

## Recurrent Neural Network
* ht - Fw(ht-1, xt)  from the old state to the new state (feedback to itself with the input)
* Language
* Image Caption 

## Detection and Segmentation
* Semantic segmentation
 * sliding window (super computational expensive)
 * Fully convolutional layer (The training data is very expensive)
   * design network as a bunch of convolutional layers, with downsampling and **upsampling** inside the network 
     * upsampling: "Unpooling" "Transpose Convolution"

* Classification + Localization
  * Treate localization as a regression problem - multi-task loss

* Object Detection
  * Sliding Window
  * Region Proposal - apply CNN to each region - SVM (R-CNN)
    * 先生成大量Region
    * 针对每个Region做SVMs+BBox Regression
  * Fast R-CNN - "Rol Pooling"
    * 先做一次卷积
    * 再对卷积的输出做Selective Search，生成大量的region
    * （ROI）分不同区域Max pooling
    * Dense
    * Softmax + BBox Regression
  * **Faster R-CNN - make CNN do proposal (Region based)**
    * 先做一次卷积
    * Region Proposal Network（RPN）—— 对任何一个Pixel生成N个大小形状不一样的Region
    * 每一个Region做一个Softmax + BBox regression看是不是我们感兴趣的region
    * ROI Pooling
    * Dense -> Softmax + BBox reg
    
  * **YOLO** 
  
  * **SSD** Single Shot Multishot Detection
    * 先做一次卷积
    * 对任何一个Pixel生成N个大小不一样的Region
    * Softmax +　BBox Reg做 N+1 Classification（看一下是不是一个Objecty以及是哪一个物体）
    * 再做一层Multiscale
    
    1. 默认边界框（锚框anchor box）
    给定大小s in (0,1]，生成的边框形状是ws * hs
    给定比例 r > 0，生成的边框形状是 w(r)^(1/2) * h/(r)^(1/2)
    
    在采集样本的时候n个大小和m个比例，简化计算我们只生成n+m-1
    * size[i] 和 rationa[0] if i <= n
    * size[0] 和 ration[i-n] if i > n
    
    2. 定义Sofemax and BBox Regression
    3. 预测物体类别
    4. 预测边框
    5. 合并多层的预测输出
    6. 减半模块
    7. 主体网络
    8. 完整的模型
    Tutorial from [GLUNO](http://zh.gluon.ai/chapter_computer-vision/ssd.html#)
    
* Instance Segmentation
  * Mask R-CNN
