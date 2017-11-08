# gp.py

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import os
import sys

# Perform gaussian process regression
# mode = 'multi'  : several trails of multiple subjects
# mode = 'single' : one single trail of one subject
def gpr(observed_data_list, mode='multi', optimized=True):
    # We divide the whole observed data into training data and test data
    # first 0 is the 1st x_clean, second 0 is the 1st marker (another example: 49 means the 50th marker)
    observed_data = observed_data_list[0][0]
    numAll = len(observed_data)
    ratio = 0.1
    X_train, y_train, X_test, y_test = [],[],[],[]
    np.random.seed(0)
    for i in range(numAll):
        rnd = np.random.random()
        if rnd < ratio:
            y_test.append(observed_data[i])
            X_test.append(i)
        else:
            y_train.append(observed_data[i])
            X_train.append(i)

    X_all = np.atleast_2d(np.asarray(range(numAll))).T
    y_all = np.atleast_2d(np.asarray(observed_data)).T
    X_train = np.atleast_2d(np.asarray(X_train)).T
    y_train = np.atleast_2d(np.asarray(y_train)).T
    X_test = np.atleast_2d(np.asarray(X_test)).T
    y_test = np.atleast_2d(np.asarray(y_test)).T

    # bad param: RBF(length_scale=0.01, length_scale_bounds=(1e-2, 1)) + ...
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=10, length_scale_bounds=(1e-2, 1e3)) \
             + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-3)
    gp.fit(X_train, y_train)
    print("\nLearned kernel: %s" % gp.kernel_)

    if mode == 'multi' and optimized:
        y_pred, sigma = gp.predict(X_all, return_std=True)
        plt.plot(X_all, y_all, 'r-', label=u'Observations')
        # 'zorder' makes the prediction line stay on top of the other lines
        # https://stackoverflow.com/questions/37246941/specifying-the-order-of-matplotlib-layers
        plt.plot(X_all, y_pred, 'b-', markersize=10, label=u'Prediction', zorder=10)
        for i in range(len(observed_data_list)):
            zorder = i / 100.0
            X_all1 = np.atleast_2d(np.asarray(range(len(observed_data_list[i][0])))).T
            y_all1 = np.atleast_2d(np.asarray(observed_data_list[i][0])).T
            plt.plot(X_all1, y_all1, 'r-', markersize=10, zorder=zorder)
        plt.xlabel('Time Frames')
        plt.ylabel('x-coordinate values of Marker 0')
        plt.legend(loc='lower right')
        plt.show()

    if mode == 'single':
        y_pred, sigma = gp.predict(X_test, return_std=True)
        plt.plot(X_test, y_test, 'r.', markersize=10, label=u'Observations')
        plt.plot(X_test, y_pred, 'b-', label=u'Prediction')
        plt.fill_between(X_test.ravel(), y_pred.ravel() - 1.9600 * sigma, y_pred.ravel() + 1.9600 * sigma, color='darkorange', alpha=0.2)
        plt.xlabel('Time Frames')
        plt.ylabel('x-coordinate values of Marker 0')
        plt.legend(loc='lower right')
        plt.show()

    if not optimized:
        # We only study the non-optimized for single trail
        gp_no = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, optimizer=None)
        gp_no.fit(X_train, y_train)
        y_pred, sigma = gp_no.predict(X_test, return_std=True)
        plt.plot(X_test, y_test, 'r.', markersize=10, label=u'Observations')
        plt.plot(X_test, y_pred, 'b-', label=u'Prediction')
        plt.fill_between(X_test.ravel(), y_pred.ravel() - 1.9600 * sigma, y_pred.ravel() + 1.9600 * sigma, color='darkorange', alpha=0.2)
        plt.xlabel('Time Frames')
        plt.ylabel('x-coordinate values of Marker 0')
        plt.legend(loc='lower right')
        plt.ylim(-1.5, 0)
        plt.show()

# We read in the file and output the corresponding vector
# x_clean, y_clean, z_clean have dimension 50x(number of timeframe with valid data)
def read_in_data(file):
    x_idx = range(11, 11+50*4, 4) # '0_x' starts at idx 11
    y_idx = range(12, 12+50*4, 4) # '0_y' starts at idx 12
    z_idx = range(13, 13+50*4, 4) # '0_z' starts at idx 13
    c_idx = range(14, 14+50*4, 4) # '0_c' starts at idx 14
    x_matrix = np.zeros((50, 1030))
    y_matrix = np.zeros((50, 1030))
    z_matrix = np.zeros((50, 1030))
    c_matrix = np.zeros((50, 1030))
    f = open(file)
    for line_num, line in enumerate(f):
        if len(line.strip()) > 0 and line_num > 0:
            fields = line.split(",")
            x_matrix[:, line_num-1] = np.transpose(map(float, np.take(fields, x_idx)))
            y_matrix[:, line_num-1] = np.transpose(map(float, np.take(fields, y_idx)))
            z_matrix[:, line_num-1] = np.transpose(map(float, np.take(fields, z_idx)))
            c_matrix[:, line_num-1] = np.transpose(map(float, np.take(fields, c_idx)))

    x_clean = []
    y_clean = []
    z_clean = []
    for num_row, crow in enumerate(c_matrix):
        # get a list of true/false boolean values based on whether greater or smaller than 0
        bool_list = np.greater(crow, np.zeros(len(crow)))
        if not all(bool_list):
            # get indices based on boolean values
            bool_list_idx = [i for i, x in enumerate(bool_list) if x]
            x_clean.append(x_matrix[num_row, :][bool_list_idx])
            y_clean.append(y_matrix[num_row, :][bool_list_idx])
            z_clean.append(z_matrix[num_row, :][bool_list_idx])
        else:
            # if all values are valid, then we directly append
            x_clean.append(x_matrix[num_row,:])
            y_clean.append(y_matrix[num_row,:])
            z_clean.append(z_matrix[num_row,:])
    return (x_clean, y_clean, z_clean)


def generate_observed_data_list(path='data_GP/', one_subject=None):
    """

    :param path:
    :param one_subject: Whether we only look at one subject, which essentially five csv files or we look at all subjects.
    :return:
    """
    observed_data_list_x = []
    observed_data_list_y = []
    observed_data_list_z = []
    def listdir_nohidden(path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f
    directories = listdir_nohidden(path)
    for dir in directories:
        if one_subject == dir:
            path2dir = os.path.abspath(path + dir)
            for root, dirs, files in os.walk(path2dir):
                for file in files:
                    if file.endswith(".csv"):
                        print(os.path.join(root, file))
                        x_clean, y_clean, z_clean = read_in_data(os.path.join(root, file))
                        observed_data_list_x.append(x_clean)
                        observed_data_list_y.append(y_clean)
                        observed_data_list_z.append(z_clean)
            break
        elif one_subject == None:
            path2dir = os.path.abspath(path + dir)
            for root, dirs, files in os.walk(path2dir):
                for file in files:
                    if file.endswith(".csv"):
                        print(os.path.join(root, file))
                        x_clean, y_clean, z_clean = read_in_data(os.path.join(root, file))
                        observed_data_list_x.append(x_clean)
                        observed_data_list_y.append(y_clean)
                        observed_data_list_z.append(z_clean)
        # if one_subject == 'AG' and one_subject == dir:
        #     break
    return (observed_data_list_x, observed_data_list_y, observed_data_list_z)


if __name__ == "__main__":
    # print observed_data_list_x[0][0]
    if len(sys.argv) >= 2:
        system_to_run = sys.argv[1]
    else:
        system_to_run = 'M'
    if system_to_run == 'S':
        observed_data_list_x, observed_data_list_y, observed_data_list_z = generate_observed_data_list()
        # look at only one single trail of one marker and learning the hyperparameters
        gpr(observed_data_list_x, 'single')
    elif system_to_run == 'M':
        observed_data_list_x, observed_data_list_y, observed_data_list_z = generate_observed_data_list()
        # Fitting trails over subjects and learning the hyperparameters
        gpr(observed_data_list_x)
    elif system_to_run == 'SN':
        observed_data_list_x, observed_data_list_y, observed_data_list_z = generate_observed_data_list()
        # We study the single trail without learning the hyperparameters
        gpr(observed_data_list_x, 'single', False)
    elif system_to_run == 'SUB':
        # We study the hypothesis states in the assignment. We use one trail of a subject to train the model
        # and we then plot all trails of the same maker of the same subject
        observed_data_list_x, observed_data_list_y, observed_data_list_z = generate_observed_data_list(one_subject='KMA')
        gpr(observed_data_list_x)
    else:
        raise Exception("Pass in either M (Multiple trails), S (Single trail), SN (Single trail, hyperparameter)"
                        "SUB (All trails with one subject) to run the appropriate system")
