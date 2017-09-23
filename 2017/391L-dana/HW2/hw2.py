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
## - Run the code with the command: python3 hw2.py

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg as la
import scipy.io as spio
#import bigfloat

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
    #mixed = np.zeros(shape=(m,original.shape[1]))
    picked_signals_to_mix = np.random.randint(original.shape[0], size=num2use)
    U = original[picked_signals_to_mix,:]  # with dimension num2use x t
    A = np.random.rand(m, num2use)
    X = np.dot(A, U)
    return picked_signals_to_mix, X

def ica(X, num2use):
    """ICA algorithm

    INPUT:
    - X       : the mixture sounds matrix
    - num2use : the number of the signals in the original sounds matrix to generate the X

    OUTPUT:
    - W : the recovered signals
    """
    def g(x):
        return 1/(1+np.exp(-x))

    #W = np.random.rand(num2use, X.shape[0])
    W = np.random.uniform(0,1,[num2use, X.shape[0]])
    numIter = 100           # termination condition
    iter = 0  
    eta = 0.1  # learning rate  
    while True:
        iter += 1
        if iter % 1000 == 0:
            eta *= 0.1
        print ("iter: " + str(iter))
        Y = np.dot(W, X)
        g = np.vectorize(g)
        Z = g(Y)
        delta_W = eta*np.dot((np.identity(num2use) + np.dot((1-2*Z), Y.T)),W)
        W = W + delta_W
        if (iter >= numIter):
            break
    return np.dot(W, X)

def plot(original, mixed, recovered_signals):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
    n = original.shape[0]
    for i in range(n):
        ax0.plot(original[i,:] / np.linalg.norm(original[i,:]))
        ax2.plot(recovered_signals[i,:] / np.linalg.norm(recovered_signals[i,:]))
    m = mixed.shape[0]
    for i in range(m):
        ax1.plot(mixed[i,:] / np.linalg.norm(mixed[i,:]))
    ax0.set_title('Original sound signals')
    ax2.set_title('Recovered sound signals')
    ax1.set_title('Mixed sound signals')
    # Tweak spacing between subplots to prevent labels from overlapping
    fig.subplots_adjust(hspace=0.3)
    plt.savefig('result.png')
    plt.show()
    plt.close()

def sounds():
    ## Read in the data file
    mat = spio.loadmat('sounds.mat')

    # 5x44000 matrix with dtype little-endian 64-bit floating point number
    # Each row represents a source signal
    sounds = mat['sounds']
    # number of original sound signals (i.e., 5)
    n = sounds.shape[0]
    # the length of each original sound (i.e., 44000)
    t = sounds.shape[1]
    # number of mixed sound signals
    m = 3
    # number of the original sound signals that are used to generate the mixed sound signals
    num2use = 2
    picked_signals_to_mix, X = mixSignal(sounds, m, num2use)
    recovered_signals = ica(X, num2use)
    plot(original[picked_signals_to_mix,:], X, recovered_signals)

def icatest():
    ## Read in the data file
    mat = spio.loadmat('icaTest.mat')
    # original sounds signal with dimension 3x40
    U = mat['U']
    # dimension 3x3
    A = mat['A']
    X = np.dot(A,U)
    recovered_signals = ica(X, U.shape[0])
    plot(U, X, recovered_signals)

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
