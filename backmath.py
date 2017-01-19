import numpy as np

"""
Utility methods for mathematical operations performed in the
context of back propagation.

__author__=Daniel Gregory
"""


def sigmoid(zs):
    """
    Return the sigmoid function of zs, where zs is
    either a numpy array or a number.
    For example:
        sigmoid(np.asarray([1,2,3]))
        sigmoid(3)
    will both work.
    """
    return 1.0 / (1.0 + np.exp(-zs))


def inverse_sigmoid(y):
    """
    Calculate the inverse of y = sigmoid(z). Provided for
    completeness and to aid with testing.
    """
    return np.log(y / (1 - y))


def sigmoid_derivative(z):
    """
    Calculate the derivative of the sigmoid function.
    """
    sig = sigmoid(z)
    return sig * (1 - sig)


def within_precision(old_value, new_value, precision):
    """
    Tests whether the new_value is within +/- epsilon of
    the old_value. When this is the case we return true,
    otherwise false.
    """
    return abs(new_value - old_value) < precision

