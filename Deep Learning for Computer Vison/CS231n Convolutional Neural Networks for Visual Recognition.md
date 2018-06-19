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

 
