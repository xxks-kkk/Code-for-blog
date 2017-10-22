# adagrad_trainer.py

from utils import *
import numpy as np


# Wrapper for using AdaGrad as the optimizer. AdagradTrainer wraps a weight vector and applies the custom
# AdaGrad update using second moments of features to make custom step sizes. This version incorporates L1
# regularization: while this regularization should be applied to squash the feature vector on every gradient update,
# we instead evaluate the regularizer lazily only when the particular feature is touched (either by gradient update
# or by access). approximate lets you turn this off for faster access, but regularization is now applied
# somewhat inconsistently.
# See section 5.1 of http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf for more details
class AdagradTrainer(object):
    # init_weights: a numpy array of the correct dimension, usually initialized to 0
    # lamb: float lambda constant for the regularizer. Values above 0.01 will often cause all features to be zeroed out.
    # eta: float step size. Values from 0.01 to 10 often work well.
    # approximate: turns off gradient updates on access, only uses them when weights are written to.
    #    So regularization is applied inconsistently, but it makes things faster.
    def __init__(self, init_weights, lamb=1e-5, eta=1.0, approximate=False):
        self.weights = init_weights
        self.lamb = lamb
        self.eta = eta
        self.approximate = approximate
        self.curr_iter = 0
        self.last_iter_touched = [0 for i in xrange(0, self.weights.shape[0])]
        self.diag_Gt = np.zeros_like(self.weights, dtype=float)

    # Take a sparse representation of the gradient and make an update, normalizing by the batch size to keep
    # hyperparameters constant as the batch size is varied
    # gradient: Counter
    # batch_size: integer
    def apply_gradient_update(self, gradient, batch_size):
        batch_size_multiplier = 1.0 / batch_size
        self.curr_iter += 1
        for i in gradient.keys():
            xti = self.weights[i]
            # N.B.We negate the gradient here because the Adagrad formulas are all for minimizing
            # and we're trying to maximize, so think of it as minimizing the negative of the objective
            # which has the opposite gradient
            # See section 5.1 in http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf for more details
            # eta is the step size, lambda is the regularization
            gti = -gradient.get_count(i) * batch_size_multiplier
            old_eta_over_Htii = self.eta / (1 + np.sqrt(self.diag_Gt[i]))
            self.diag_Gt[i] += gti * gti
            Htii = 1 + np.sqrt(self.diag_Gt[i])
            eta_over_Htii = self.eta / Htii
            new_xti = xti - eta_over_Htii * gti
            # Apply the regularizer for every iteration since touched
            iters_since_touched = self.curr_iter - self.last_iter_touched[i]
            self.last_iter_touched[i] = self.curr_iter
            self.weights[i] = np.sign(new_xti) * max(0, np.abs(new_xti) - self.lamb * eta_over_Htii - (iters_since_touched - 1) * self.lamb * old_eta_over_Htii)

    # Get the weight of feature i
    def access(self, i):
        if not self.approximate and self.last_iter_touched[i] != self.curr_iter:
            xti = self.weights[i]
            Htii = 1 + np.sqrt(self.diag_Gt[i])
            eta_over_Htii = self.eta / Htii
            iters_since_touched = self.curr_iter - self.last_iter_touched[i]
            self.last_iter_touched[i] = self.curr_iter
            self.weights[i] = np.sign(xti) * max(0, np.abs(xti) - iters_since_touched * self.lamb * self.eta * eta_over_Htii);
        return self.weights[i]

    # Scores a sparse feature vector
    # feats: list of integer feature indices
    def score(self, feats):
        i = 0
        score = 0.0
        while i < len(feats):
            score += self.access(feats[i])
            i += 1
        return score

    # Return a numpy array containing the final weight vector values -- manually calls access to force each weight to
    # have an updated value.
    def get_final_weights(self):
        for i in xrange(0, self.weights.shape[0]):
            self.access(i)
        return self.weights
