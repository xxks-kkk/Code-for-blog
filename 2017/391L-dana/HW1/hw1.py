#!/usr/bin/python3

import numpy as np
from scipy import linalg as la
import scipy.io as spio
import matplotlib.pyplot as plt

## Read in the data file
mat = spio.loadmat('digits.mat')
testLabels = mat['testLabels']
testImages = mat['testImages']
trainImages = mat['trainImages']
trainLabels = mat['trainLabels']

## Number of training images we use
M = 400

## Construct our data matrix with dimension 784xM. 784 is calculated by
## 784 = 28*28
data = trainImages[:,:,0,0:M].reshape(784,M)

## We first calculate the "average face"
mu = np.mean(data,axis=1)
data -= np.tile(mu.reshape(784,1),M).astype(data.dtype)

## Now we calculate the covariance matrix
# C = np.cov(data)
C = np.dot(data, np.transpose(data))

## Compute the eigenvalues and eigenvectors and sort into descending order
evals, evecs = np.linalg.eig(C)
indices = np.argsort(evals) # is in ascending order
indices = indices[::-1]     # change to descending order
evecs = evecs[:,indices]
evals = evals[indices]

## Set nRedDim (i.e., k)
nRedDim = 1

evecs = evecs[:,:nRedDim]

## Produce the new data matrix
x = np.dot(np.transpose(evecs), data)

## Compute the original data
y = np.dot(evecs,x)+np.tile(mu.reshape(784,1),M)
# plt.imshow(y[:,1].reshape(28,28).real) 
