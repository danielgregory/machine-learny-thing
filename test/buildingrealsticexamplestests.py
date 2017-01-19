import unittest

import numpy as np

from backmath import sigmoid
from ml import learn
from network import Network


class RealisticExamples(unittest.TestCase):
    def test_me(self):
        """
        Very simple first case: one input, one output, one node test.
        """
        # begin with random values for the weights and biases.
        weights = [np.asarray([1])]
        biases = [np.asarray([2])]
        network = Network(weights, biases)

        # these are the weights and biases we are aiming to have
        # in the trained network.
        trained_w = 5
        trained_b = 6
        training_data = generate_training_data(trained_w, trained_b)

        trained_network = learn(network, training_data, 100)

        self.assertAlmostEqual(trained_w, trained_network.weights[0][0])


def generate_training_data(trained_w, trained_b):
    return [(x, sigmoid(trained_w * x + trained_b)) for x in range(100)]
