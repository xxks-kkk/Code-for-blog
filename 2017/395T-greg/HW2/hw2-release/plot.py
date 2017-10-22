# Plot the Greedy parser and Global Beam-search parser experimentation data

import matplotlib.pyplot as plt
import numpy as np

def plot():
    eng_greedy = np.array([72.08, 74.42, 75.34, 75.72, 75.94, 76.01, 75.90, 75.93, 75.90, 75.78, 78.17, 78.17, 78.19, 78.27, 78.25, 78.21, 78.22, 78.27, 78.25, 78.22, 
                            78.66, 78.57, 78.59, 78.60, 78.61, 78.60, 78.61, 78.64, 78.65, 78.63])
    eng_beam    = np.array([74.16, 71.61, 75.39, 75.17, 73.96, 75.21, 73.97, 74.95, 74.89, 75.67])
    
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 18,
           }
    
    plt.plot(eng_greedy, label='Greedy parser')
    plt.plot(eng_beam, label='Global Beam-search parser')
    plt.title('UAS score vs. number of epoches', fontdict=font)
    plt.xlabel('number of epoches', fontdict=font, fontsize=18)
    plt.ylabel('UAS score', fontdict=font, fontsize=18)
    plt.legend(loc='lower right', fontsize='17')
    plt.savefig('filename.png', dpi = 500)
    plt.show()

if __name__ == "__main__":
    plot()
