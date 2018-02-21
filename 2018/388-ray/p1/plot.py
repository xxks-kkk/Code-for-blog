# Plot the Greedy parser and Global Beam-search parser experimentation data

import matplotlib.pyplot as plt
import numpy as np

def plot():
    y = np.array([157.54, 139.45, 131.02, 127.07, 126.11, 127.81, 132.57, 142.00, 161.54])
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 18,
           }
    
    plt.plot(x,y, label='Word perplexiy')
    plt.title('Word perplexity vs. weights of bigram in bidirectional model', fontdict=font)
    plt.xlabel('Weights of bigram in birdirectional $\lambda_1$', fontdict=font, fontsize=18)
    plt.ylabel('Word perplexity on test set of Penn Treebank', fontdict=font, fontsize=18)
    plt.legend(loc='lower right', fontsize='17')
    plt.savefig('filename.png', dpi = 500)
    plt.show()

if __name__ == "__main__":
    plot()
