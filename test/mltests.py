import unittest

import numpy as np
import backmath as bm

from ml import _network_has_converged, learn, learn_stuff
from network import Network


class MLTests(unittest.TestCase):
    def test__network_has_converged(self):
        updated_network = Network([np.asarray([1, 1]), np.asarray([1, 1])],
                                  [np.asarray([2, 2])])

        previous_weights = [np.asarray([1.5, 1.5]), np.asarray([1.5, 1.5])]
        previous_biases = [np.asarray([2.2, 2.2])]

        has_converged = _network_has_converged(previous_weights, previous_biases, updated_network, 1)

        self.assertTrue(has_converged)

    def test__network_has_not_converged(self):
        updated_network = Network([np.asarray([1, 1]), np.asarray([1, 1])],
                                  [np.asarray([2, 2])])

        previous_weights = [np.asarray([1.5, 1.5]), np.asarray([1.5, 1.5])]
        previous_biases = [np.asarray([2.2, 2.2])]

        has_converged = _network_has_converged(previous_weights, previous_biases, updated_network, 0.1)

        self.assertFalse(has_converged)

    def test_learning_in_one_one_network(self):
        initial_weights = [np.asarray([0.4]), np.asarray([0.1])]
        true_weights = [np.asarray([0.5]), np.asarray([0.2])]

        initial_biases = [np.asarray([0.2]), np.asarray([0.4])]
        true_biases = [np.asarray([0.3]), np.asarray([0.6])]

        network = Network(initial_weights, initial_biases)

        training_data = [(0, 0.5535163485),
                         (0.1, 0.5637451207),
                         (0.2, 0.5732348081),
                         (0.3, 0.5820319469),
                         (0.4, 0.5901830653),
                         (0.5, 0.5977335206),
                         (0.6, 0.6047267307),
                         (0.7, 0.61120371),
                         (0.8, 0.6172028311)]

        trained_network = learn_stuff(network, training_data, 10000, 0.01)
        self.assertAlmostEqual(true_weights, trained_network.weights[0][0])
        self.assertAlmostEqual(true_biases, trained_network.biases[0])

    def another_thing(self):
        training_data = []
        w_1 = 0.5
        w_2 = 0.2
        b_1 = 0.3
        b_2 = 0.6

        initial_weights = [np.asarray([0.4]), np.asarray([0.1])]
        true_weights = [np.asarray([0.5]), np.asarray([0.2])]

        initial_biases = [np.asarray([0.2]), np.asarray([0.4])]
        true_biases = [np.asarray([0.3]), np.asarray([0.6])]

        network = Network(initial_weights, initial_biases)

        for x in range(0, 20, 0.1):
            a_2 = w_2 * bm.sigmoid(w_1 * x + b_1) + b_2
            result = bm.sigmoid(a_2)
            training_data.append((x, result))

        trained_network = learn(network, training_data, 1000000, 0.01)
        self.assertAlmostEqual(true_weights, trained_network.weights[0][0])
        self.assertAlmostEqual(true_biases, trained_network.biases[0])
