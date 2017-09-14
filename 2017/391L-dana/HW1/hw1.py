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
M = 4000

## Set nRedDim (i.e., k)
nRedDim = 1

## Construct our data matrix with dimension 784xM. 784 is calculated by
## 784 = 28*28
data = trainImages[:,:,0,0:M].reshape(784,M).astype('float')

## We first calculate the "average face"
mu = np.mean(data,axis=1)
data -= np.tile(mu.reshape(784,1),M).astype(data.dtype)
#standard = np.std(data,axis=1)
#data_normalized = np.divide(data, (standard+0.0001).reshape(784,1))

## Now we calculate the covariance matrix
#C = np.cov(data)
C = np.dot(data, data.T)
#C = np.dot(data_normalized, np.transpose(data_normalized)) * (1/M)

## Compute the eigenvalues and eigenvectors and sort into descending order
evals, evecs = np.linalg.eig(C)
indices = np.argsort(evals) # is in ascending order
indices = indices[::-1]     # change to descending order
evecs = evecs[:,indices]
evals = evals[indices]

## Try to determine the number of principal component K
U,S,V = np.linalg.svd(C)
print('X  Rate')
for x in range(1,M):
    rate = S[:x].sum()/S.sum()
    print('{0:2d} {1:3f}'.format(x,rate))
    if rate >= 0.99:
        nRedDim = x
        break

## select the top K eigenvectors
evecs = evecs[:,:nRedDim]

## normalize eigenvectors
evecs = np.divide(evecs, np.linalg.norm(evecs, axis=0))

## Produce the new data matrix
x = np.dot(np.transpose(evecs), data)

## Compute the original data
y = np.dot(evecs,x)+np.tile(mu.reshape(784,1),M)

plt.imshow(y[:,1].reshape(28,28).real) 
plt.show()
