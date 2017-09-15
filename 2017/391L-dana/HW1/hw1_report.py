#!/usr/bin/python3

## HW1 | Principle Component Analysis (PCA) and K-Nearest Neighbour
## Clustering
##
## AUTHOR: Zeyuan Hu <iamzeyuanhu@utexas.edu | EID: zh4378>
##
## INSTRUCTIONS
## ------------
##
## - Please make sure you have the python package "numpy", "scipy", "matplotlib",
##   and "sklearn" installed in your environment
## - Put the data file "digits.mat" under the same directory with this source
##   code "hw1.py"
## - Run the code with the command: python3 hw1.py

import numpy as np
from scipy import linalg as la
import scipy.io as spio
import scipy.stats as spstats
import matplotlib.pyplot as plt
from sklearn import neighbors

def knn(testImages, testLabels, data, trainLabels, evecs):
    """ K nearest neighbor for our task

    - This function returns the accuracy of our classification on the test data
    - The arguments to this function should come from the exact corresponding
      argument name in our main() function
    """

    ## KNN classification
    knn=neighbors.KNeighborsClassifier()
    num_testImages = testImages.shape[3]
    testImages_use = testImages[:,:,0,0:num_testImages].reshape(784, num_testImages)
    test_mu = np.mean(testImages_use,axis=1)
    testImages_normalize = testImages_use -  np.tile(test_mu.reshape(784,1),num_testImages).astype(testImages_use.dtype)
    testImages_es = np.dot(evecs.T,testImages_normalize)
    testLabels_arr = testLabels.squeeze()
    num_trainImages = data.shape[1]
    trainImages_es = np.dot(np.transpose(evecs), data)
    trainLabels_arr = np.squeeze(trainLabels)

    knn.fit(trainImages_es.real.T, trainLabels_arr[:num_trainImages])
    pred = knn.predict(testImages_es.real.T)

    accuracy = np.sum(np.equal(pred,testLabels_arr[:num_testImages])) / float(pred.shape[0])

    return accuracy


def nRedDim2Accuracy(testImages, testLabels, trainImages, trainLabels):
    """ Experiment 2: Examine the relationship between the accuracy of classification
    and the number of eigenvectors we keep (i.e. nRedDim)

    - The arguments to this function should come from the exact corresponding
      argument name in our main() function
    """
    numIter = 30
    M = 1000
    accuracy_arr = np.zeros(numIter)

    ## Construct our data matrix with dimension 784xM. 784 is calculated by
    ## 784 = 28*28
    data = trainImages[:,:,0,0:M].reshape(784,M).astype('float')
    testImages = testImages.astype('float')

    ## We first calculate the "average face"
    mu = np.mean(data,axis=1)
    data -= np.tile(mu.reshape(784,1),M).astype(data.dtype)

    ## Andrew Ng's way for preprocessing data
    # standard = np.std(data,axis=1)
    # data_normalized = np.divide(data, (standard+0.0001).reshape(784,1))

    ## Now we calculate the covariance matrix
    C = np.dot(data, data.T)

    ## Andrew Ng's way for calculating the covariance matrix
    # C = np.dot(data_normalized, np.transpose(data_normalized)) * (1/M)

    ## Compute the eigenvalues and eigenvectors and sort into descending order
    evals, evecs = np.linalg.eig(C)
    indices = np.argsort(evals)     # is in ascending order
    indices = indices[::-1]         # change to descending order
    evecs = evecs[:,indices]
    evals = evals[indices]
    
    for x in range(1,numIter):
        nRedDim = x

        ## select the top nRedDim eigenvectors
        evecs2 = evecs[:,:nRedDim]
        ## normalize eigenvectors
        evecs2 = np.divide(evecs2, np.linalg.norm(evecs2, axis=0))
        accuracy_arr[x] = knn(testImages, testLabels, data, trainLabels, evecs2)
        print("{}, {}".format(x, accuracy_arr[x]))

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16,
           }

    plt.plot(range(numIter), accuracy_arr)
    plt.title('Classification accuracy vs. number of eigenvectors', fontdict=font)
    plt.xlabel('number of eigenvectors', fontdict=font)
    plt.ylabel('accuracy', fontdict=font)
    plt.subplots_adjust(left=0.15)
    plt.savefig('nRedDim2Accuracy.png')
    plt.show()
    plt.close()

    
def numTrain2Accuracy(testImages, testLabels, trainImages, trainLabels):
    """ Experiment 1: Examine the relationship between the accuracy of classification
    and the number of training points

    - The arguments to this function should come from the exact corresponding
      argument name in our main() function
    """
    numIter = 420
    accuracy_arr = np.zeros(numIter)

    ## The reason we start from 5 is that KNN algorithm takes n_neighbors = 5
    ## as default and the algorithm requires n_neighbors <= n_samples
    for x in range(5, numIter): 
        ## Number of training images we use
        M = x

        ## Set nRedDim
        nRedDim = 20

        ## Construct our data matrix with dimension 784xM. 784 is calculated by
        ## 784 = 28*28
        data = trainImages[:,:,0,0:M].reshape(784,M).astype('float')
        testImages = testImages.astype('float')

        ## We first calculate the "average face"
        mu = np.mean(data,axis=1)
        data -= np.tile(mu.reshape(784,1),M).astype(data.dtype)

        ## Andrew Ng's way for preprocessing data
        # standard = np.std(data,axis=1)
        # data_normalized = np.divide(data, (standard+0.0001).reshape(784,1))

        ## Now we calculate the covariance matrix
        C = np.dot(data, data.T)

        ## Andrew Ng's way for calculating the covariance matrix
        # C = np.dot(data_normalized, np.transpose(data_normalized)) * (1/M)

        ## Compute the eigenvalues and eigenvectors and sort into descending order
        evals, evecs = np.linalg.eig(C)
        indices = np.argsort(evals)     # is in ascending order
        indices = indices[::-1]         # change to descending order
        evecs = evecs[:,indices]
        evals = evals[indices]
        ## select the top nRedDim eigenvectors
        evecs2 = evecs[:,:nRedDim]
        ## normalize eigenvectors
        evecs2 = np.divide(evecs2, np.linalg.norm(evecs2, axis=0))
        accuracy_arr[x] = knn(testImages, testLabels, data, trainLabels, evecs2)
        print("{}, {}".format(x, accuracy_arr[x]))
        
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16,
           }

    plt.plot(range(numIter), accuracy_arr)
    plt.title('Classification accuracy vs. number of training points', fontdict=font)
    plt.xlabel('number of training points', fontdict=font)
    plt.ylabel('accuracy', fontdict=font)
    plt.subplots_adjust(left=0.15)
    plt.savefig('numTrain2Accuracy.png')
    plt.show()
    plt.close()

    
def main():
    ## Read in the data file
    mat = spio.loadmat('digits.mat')
    testLabels = mat['testLabels']
    testImages = mat['testImages']
    trainImages = mat['trainImages']
    trainLabels = mat['trainLabels']

    # #x = np.dot(np.transpose(evecs), data)R
    # x = np.dot(np.transpose(evecs), testImages[:,:,0,0:M].reshape(784,M).astype('float'))
    # ## Compute the original data
    # y = np.dot(evecs,x)+np.tile(mu.reshape(784,1),M)

    # plt.imshow(y[:,0].reshape(28,28).real) 
    # plt.show()

    ## Experiment 1: Examine the relationship between the accuracy of classification
    ## and the number of training points
    numTrain2Accuracy(testImages, testLabels, trainImages, trainLabels)    
    
    ## Experiment 2: Examine the relationship between the accuracy of classification
    ## and the number of eigenvectors we keep (i.e. nRedDim)    
    nRedDim2Accuracy(testImages, testLabels, trainImages, trainLabels)

if __name__ == "__main__":
    main()

