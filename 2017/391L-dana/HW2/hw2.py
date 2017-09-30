#!/usr/bin/python3

## HW2 | Independent Component Analysis (ICA)
##
## AUTHOR: Zeyuan Hu <iamzeyuanhu@utexas.edu | EID: zh4378>
##
## INSTRUCTIONS
## ------------
##
## - Please make sure you have the python package "numpy", "scipy", "matplotlib",
##   installed in your environment
## - Put the data file "sounds.mat" under the same directory with this source
##   code "hw2.py"
## - Run the code with the command:
##   - python3 hw2.py ICATEST
##   - python3 hw2.py SOUNDS

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as la
import scipy.io as spio

def mixSignal(original, m, num2use):
    """Generate the mixture matrix with dimension mxt
    from sounds matrix with dimension nxt

    INPUT:
    - original  : original sounds matrix with dimension mxt
    - m         : number of the mixed signals want to generate, which is the num of rows in mixed matrix
    - num2use   : number of the original signals you want to use to generate the mixture matrix

    OUTPUT:
    - mixed     : the mixture sounds matrix
    """
    picked_signals_to_mix = np.random.randint(original.shape[0], size=num2use)
    U = original[picked_signals_to_mix,:]  # with dimension num2use x t
    A = np.random.rand(m, num2use)
    X = np.dot(A, U)
    return picked_signals_to_mix, X

def ica(X, num2use, eta):
    """ICA algorithm

    INPUT:
    - X       : the mixture sounds matrix
    - num2use : the number of the signals in the original sounds matrix to generate the X
    - eta     : the learning rate

    OUTPUT:
    - W : the recovered signals
    """
    W = np.random.uniform(0,1,[num2use, X.shape[0]])
    numIter = 1000000           # termination condition1
    threshold = 1e-9
    iter = 0  
    while True:
        iter += 1
        if iter % 1000 == 0:
            eta *= 0.1
        print ("iter: " + str(iter))
        Y = W*X
        Z = np.divide(1.0, (1+np.exp(-Y)))
        delta_W = eta*(np.identity(num2use) + (1-2*Z)*Y.T)*W
        ndelta = np.linalg.norm(delta_W)
        if (ndelta < threshold or iter >= numIter):
            break
        print("delta_W: ")
        print(delta_W)
        W = W + delta_W
    return W*X

def plot(original, mixed, recovered_signals, img_name):
    """ Draw the picture in a compact way: original signals,
    mixed signals, and recovered signals all in one picture.

    Also, draw the different signals separetly for easier comparison.
    """

    # Draw the image in a compact way

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 12,
           }
        
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
    n = original.shape[0]
    for i in range(n):
        # +i because we don't want all the signals clustered together
        ax0.plot(np.squeeze(np.asarray(original[i,:])) + i)        
        ax2.plot(np.squeeze(np.asarray(recovered_signals[i,:])) + i)
    m = mixed.shape[0]
    for i in range(m):
        ax1.plot(np.squeeze(np.asarray(mixed[i,:])) + i)        
    ax0.set_title('Original sound signals', fontdict=font)
    ax2.set_title('Recovered sound signals', fontdict=font)
    ax1.set_title('Mixed sound signals', fontdict=font)
    # Tweak spacing between subplots to prevent labels from overlapping
    fig.subplots_adjust(hspace=0.3)
    plt.savefig(img_name)
    plt.show()
    plt.close()

    # Draw the images spearately

    # n = original.shape[0]
    # for i in range(n):
    #     plt.plot(np.squeeze(np.squeeze(np.asarray(original[i,:])))+i)
    # plt.title('Classification accuracy vs. number of eigenvectors', fontdict=font)
    # plt.xlabel('number of eigenvectors', fontdict=font)
    # plt.ylabel('accuracy', fontdict=font)
    # plt.savefig('nRedDim2Accuracy.png')
    # plt.show()
    # plt.close()

def rescaleSignals(signals, minVal, maxVal):
    """ This function is used for normalizing signals because scale doesn't
    change result according to Prof. Andrew Ng's lecture note
    """
    numRows, numCols = signals.shape
    newSignals = np.zeros((numRows, numCols))
    rowMaxVals = np.amax(signals, 1)
    rowMinVals = np.amin(signals, 1)
    k = (maxVal - minVal) / (rowMaxVals - rowMinVals)    
    for i in range(numRows):
        newSignals[i,:] = (signals[i,:] - np.asmatrix(np.repeat(rowMinVals[i],numCols), dtype='float32'))*k.item(i) + \
                          np.asmatrix(np.repeat(minVal, numCols), dtype='float32')
    return newSignals
    
def icatest():
    ## Read in the data file
    mat = spio.loadmat('icaTest.mat')
    # original sounds signal with dimension 3x40
    U = np.asmatrix(mat['U'], dtype='float32')
    # dimension 3x3
    A = np.asmatrix(mat['A'], dtype='float32')
    X = A*U
    recovered_signals = ica(X, U.shape[0], 0.01)
    normalized_recovered_signals = rescaleSignals(recovered_signals, 0, 1)
    normalized_U = rescaleSignals(U, 0, 1)
    normalized_X = rescaleSignals(X, 0, 1)
    plot(normalized_U, normalized_X, normalized_recovered_signals, "icaTest")

def sounds():
    ## Read in the data file
    mat = spio.loadmat('sounds.mat')
    # 5x44000 matrix with dtype little-endian 64-bit floating point number
    # Each row represents a source signal
    sounds = np.asmatrix(mat['sounds'], dtype='float32')
    # number of original sound signals (i.e., 5)
    n = sounds.shape[0]
    # the length of each original sound (i.e., 44000)
    t = sounds.shape[1]
    # number of mixed sound signals
    m = 3
    # number of the original sound signals that are used to generate the mixed sound signals
    num2use = 2
    picked_signals_to_mix, X = mixSignal(sounds, m, num2use)
    recovered_signals = ica(X, num2use, 0.001)
    normalized_X = rescaleSignals(X, 0, 1)
    normalized_recovered_signals = rescaleSignals(recovered_signals, 0, 1)
    normalized_U = rescaleSignals(sounds[picked_signals_to_mix,:], 0, 1)
    # make sure we don't accidentally set both normalized_U and normalized_recovered_signals to be the same
    assert((normalized_U != normalized_recovered_signals).all()) 
    plot(normalized_U, normalized_X, normalized_recovered_signals, "sounds")
    
if __name__ == "__main__":    
    if len(sys.argv) >= 2:
        system_to_run = sys.argv[1]
    else:
        system_to_run = "ICATEST"
    if system_to_run == "ICATEST":
        icatest()
    elif system_to_run == "SOUNDS":
        sounds()
    else:
        raise Exception("Pass in either ICATEST or SOUNDS to run the ICA algorithm on icatest.mat or sounds.mat respectively")
