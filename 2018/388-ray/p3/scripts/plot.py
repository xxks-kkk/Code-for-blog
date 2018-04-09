# Plot the learning curves

import matplotlib.pyplot as plt
import numpy as np

def plot():
    random_x = np.array([1179, 2696, 4210, 5725, 7239, 8750, 10270, 11792, 13309, 14830, 16331,
                         17852, 19381, 20888, 22403, 23905, 25442, 26944, 28461, 29976])
    random_y = np.array([44.27, 62.26, 65.61, 70.22, 73.06, 74.82, 76.07, 76.09, 77.12, 77.46, 77.69,
                         77.93, 78.68, 78.69, 79.68, 79.65, 79.87, 79.74, 80.31, 80.44])
    length_x = np.array([1179, 2716, 4246, 5792, 7301, 8807, 10309, 11850, 13393, 14894,
                         16406, 17926, 19451, 20979, 22499, 24003, 25519, 27031, 28542, 30047])
    length_y = np.array([43.65, 57.38, 63.43, 67.28, 68.69, 69.77, 71.99, 72.70, 74.49,
                         74.97, 76.31, 76.05, 76.89, 77.22, 77.65, 78.56, 77.97, 77.40,
                         78.31, 78.66])
    raw_x = np.array([1179, 2694, 4206, 5753, 7264, 8766, 10299, 11833, 13363, 14871, 16376,
                      17878, 19393, 20906, 22420, 23943, 25461, 26970, 28497, 30006])
    raw_y = np.array([43.37, 63.28, 68.86, 71.01, 73.55, 74.47, 76.40, 77.22, 78.24, 78.37,
                      79.24, 80.00, 79.82, 80.55, 79.74, 80.75, 80.56, 81.34, 80.99, 81.51])
    margin_x = np.array([1179, 2700, 4218, 5733, 7235, 8747, 10263, 11774, 13298, 14808,
                         16313, 17840, 19368, 20874, 22377, 23896, 25406, 26923, 28433, 29942])
    margin_y = np.array([43.32, 61.44, 67.74, 71.43, 72.83, 74.16, 74.35, 75.83, 76.06, 77.53,
                         77.44, 77.80, 78.84, 78.50, 78.88, 79.78, 79.07, 79.43, 79.85, 79.60])
    
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 18,
           }
    
    plt.plot(random_x, random_y, label='Random')
    plt.plot(length_x, length_y, label='Length')
    plt.plot(raw_x, raw_y, label='Raw')
    plt.plot(margin_x, margin_y, label='Margin')
    plt.axvline(x=2694, linestyle='dashed', color='black') 
    plt.axvline(x=6600, linestyle='dashed', color='black')
    plt.title('Learning Curves for Active Learning', fontdict=font)
    plt.xlabel('number of training words', fontdict=font, fontsize=18)
    plt.ylabel('LAS score', fontdict=font, fontsize=18)
    plt.legend(loc='lower right', fontsize='17')
    plt.savefig('filename.png', dpi = 500)
    plt.show()

if __name__ == "__main__":
    plot()
