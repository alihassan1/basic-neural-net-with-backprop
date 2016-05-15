# Handwritten digit recognition using Neural Networks in Matlab

## Overview

This is a basic implantation of Stochastic Gradient Descent (SGD) for a feedforward neural network. Backpropagation is performed to update gradients for each neuron in the network. This implementation is tested on MNIST dataset with multiple configurations i.e. 1 to 5 hidden layers of varied sizes. The best detection rate that I have been able to achieve is 95% using a single hidden layer of 30 neurons.

Current implication has only sigmoid as an activation function. I have plans to implement ReLu to see how well it performs in this configuration. It also includes dropout regularization, which is useful for multiple hidden layers. 

## Usage
Download MNIST dataset (http://yann.lecun.com/exdb/mnist/). Note that this repository contains functions required to load MINST dataset. These function are borrowed from UFLDL tutorial (http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset)
Main.m as the name suggests, is the starter file. It is divided into multiple sections. Please update data paths in ‘Load Data’ section of the script. You can also play around with parameters such as ‘learning rate’, ‘batch size’ etc.
