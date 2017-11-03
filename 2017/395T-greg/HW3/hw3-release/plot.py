# Plot the Greedy parser and Global Beam-search parser experimentation data

import matplotlib.pyplot as plt
import numpy as np

def plot():
    # accuracy based on steps
    # eng_cnn = np.array([60.51, 60.98, 68.20, 68.67, 65.29, 65.95, 71.86, 73.36,
    #                        72.61, 72.70, 74.77, 74.67, 75.42, 73.36, 71.67, 70.64,
    #                        75.89, 75.14, 74.11, 76.92, 76.08, 74.77, 75.79, 76.83,
    #                        76.55, 76.55, 75.05, 75.61, 74.11, 75.99, 75.80, 75.99,
    #                        76.36, 76.92, 77.20, 77.58, 77.49, 77.39, 77.30, 76.74,
    #                        77.11, 77.30, 77.39, 77.96, 77.96, 77.67, 77.86, 78.33,
    #                        78.05, 78.24, 77.96, 78.24, 78.52, 78.05, 77.77, 77.67,
    #                        77.49, 78.14, 78.80, 78.14, 78.71, 78.61, 78.24, 78.61,
    #                        78.61, 78.61, 78.71, 78.33, 78.80, 78.80, 79.08, 79.54])
    eng_cnn = np.array([75.80, 76.45, 77.30, 77.96, 78.42, 77.58, 78.33, 77.86, 78.89, 78.24,
                        77.96, 77.77, 78.33, 78.05, 78.24, 77.11, 76.92, 77.39, 77.58, 78.24])

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 18,
           }
    
    plt.plot(eng_cnn, label='CNN')
    plt.title('Accuracy vs. number of epochs', fontdict=font)
    plt.xlabel('number of epochs', fontdict=font, fontsize=18)
    plt.ylabel('Accuracy', fontdict=font, fontsize=18)
    plt.legend(loc='lower right', fontsize='17')
    plt.savefig('filename.png', dpi = 500)
    plt.show()

if __name__ == "__main__":
    plot()
