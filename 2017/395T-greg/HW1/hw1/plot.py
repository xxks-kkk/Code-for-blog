# Plot the CRF experimentation data

import matplotlib.pyplot as plt
import numpy as np

def plot():
    eng_viterbi = np.array([69.40, 75.94, 72.58, 77.66, 79.43, 76.22, 79.61, 81.56, 81.02, 81.11, 84.66, 84.84, 85.29, 85.30,
                            85.32, 85.38, 85.29, 85.32, 85.43, 85.47, 85.77, 85.70, 85.65, 85.71, 85.74, 85.77])
    deu_viterbi = np.array([44.89, 48.54, 46.65, 51.87, 52.01, 51.32, 48.74, 50.01, 52.98, 52.52, 58.06, 57.59, 57.38, 57.06,
                            56.83, 56.64, 56.54, 56.57, 56.62, 56.72, 56.64, 56.63, 56.36, 56.37, 56.37, 56.30])
    eng_beam    = np.array([69.21, 74.20, 70.89, 77.97, 79.53, 77.46, 78.74, 79.09, 79.23, 81.39, 84.52, 84.95, 85.06, 85.12, 
                            85.22, 85.43, 85.37, 85.39, 85.46, 85.42, 85.66, 85.75, 85.80, 85.79, 85.79, 85.77])
    
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 12,
           }
    
    plt.plot(eng_viterbi, label='ENG, Viterbi Algorithm')
    plt.plot(deu_viterbi, label='DEU, Viterbi Algorithm')
    plt.plot(eng_beam, label='ENG, Beam Search')
    plt.title('F1 score vs. number of epoches', fontdict=font)
    plt.xlabel('number of epoches', fontdict=font)
    plt.ylabel('F1 score', fontdict=font)
    i = 20 # annotate ith point
    label = 'F1 Score: {0}'.format(eng_viterbi[i])
    plt.annotate(label, xy=(i, eng_viterbi[i]), xytext=(40, -40), 
                textcoords='offset points', ha='right', va='bottom', 
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5), 
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    plot()