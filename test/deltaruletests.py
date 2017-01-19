import unittest

import numpy as np

from ml import *


class DeltaRuleTests(unittest.TestCase):
    """

    Tests the delta rule. This corresponds to a special case where the network is
    single-layered and there are no hidden (intermediate) nodes between the input
    and output node.

    The test cases start off with a simple case and slowly become more extreme. For
    example, the untrained weight is initialised further from the true value, and
    we add a bias term.

    """

    def test_delta_rule_with_zero_bias(self):
        weights = [np.asarray([0.1])]
        biases = [np.asarray([0])]
        network = Network(weights, biases)
        # only 300 iterations. Later tests require much more.
        iterations = 300
        training_data = [(0.1, 0.5149955016),
                         (0.2, 0.5299640518),
                         (0.3, 0.5448788924),
                         (0.4, 0.5597136493),
                         (0.5, 0.5744425168),
                         (0.6, 0.5890404341),
                         (0.7, 0.6034832499),
                         (0.8, 0.6177478748),
                         (0.9, 0.6318124177)]
        trained_network = learn(network, training_data, iterations, 1)
        self.assertAlmostEqual(trained_network.weights[0][0], 0.6, None, None, 0.01)
        self.assertAlmostEqual(trained_network.biases[0][0], 0.0, None, None, 0.01)

    def test_delta_rule_with_zero_bias_weight_is_far_from_true_value(self):
        weights = [np.asarray([5])]
        biases = [np.asarray([0])]
        network = Network(weights, biases)
        # we need more iterations here to get to the true weights and biases
        iterations = 600
        training_data = [(0.1, 0.5149955016),
                         (0.2, 0.5299640518),
                         (0.3, 0.5448788924),
                         (0.4, 0.5597136493),
                         (0.5, 0.5744425168),
                         (0.6, 0.5890404341),
                         (0.7, 0.6034832499),
                         (0.8, 0.6177478748),
                         (0.9, 0.6318124177)]
        trained_network = learn(network, training_data, iterations, 1)
        self.assertAlmostEqual(trained_network.weights[0][0], 0.6, None, None, 0.01)
        self.assertAlmostEqual(trained_network.biases[0][0], 0.0, None, None, 0.01)

    def test_delta_rule_with_zero_bias_weight_is_far_from_true_value_and_negative(self):
        weights = [np.asarray([-5])]
        biases = [np.asarray([0])]
        network = Network(weights, biases)
        iterations = 12000
        training_data = [(0.1, 0.5149955016),
                         (0.2, 0.5299640518),
                         (0.3, 0.5448788924),
                         (0.4, 0.5597136493),
                         (0.5, 0.5744425168),
                         (0.6, 0.5890404341),
                         (0.7, 0.6034832499),
                         (0.8, 0.6177478748),
                         (0.9, 0.6318124177)]
        trained_network = learn(network, training_data, iterations, learning_rate=1)
        self.assertAlmostEqual(trained_network.weights[0][0], 0.6, None, None, 0.01)
        self.assertAlmostEqual(trained_network.biases[0][0], 0.0, None, None, 0.01)

    def test_delta_rule_with_bias(self):
        weights = [np.asarray([0.3])]
        biases = [np.asarray([0.1])]
        network = Network(weights, biases)
        iterations = 1200
        training_data = [(0.1, 0.7026606543),
                         (0.2, 0.7150421057),
                         (0.3, 0.7271082163),
                         (0.4, 0.7388500061),
                         (0.5, 0.7502601056),
                         (0.6, 0.7613327148),
                         (0.7, 0.7720635494),
                         (0.8, 0.7824497764),
                         (0.9, 0.7924899414)]
        trained_network = learn(network, training_data, iterations, 1)
        true_weight = 0.6
        true_bias = 0.8
        self.assertAlmostEqual(trained_network.weights[0][0], true_weight, None, None, 0.01)
        self.assertAlmostEqual(trained_network.biases[0][0], true_bias, None, None, 0.01)

    def test_delta_rule_with_bias_initial_values_further_from_truth(self):
        weights = [np.asarray([5])]
        biases = [np.asarray([6])]
        network = Network(weights, biases)
        iterations = 12000
        training_data = [(0.1, 0.7026606543),
                         (0.2, 0.7150421057),
                         (0.3, 0.7271082163),
                         (0.4, 0.7388500061),
                         (0.5, 0.7502601056),
                         (0.6, 0.7613327148),
                         (0.7, 0.7720635494),
                         (0.8, 0.7824497764),
                         (0.9, 0.7924899414)]
        trained_network = learn(network, training_data, iterations, 1)
        true_weight = 0.6
        true_bias = 0.8
        self.assertAlmostEqual(trained_network.weights[0][0], true_weight, None, None, 0.01)
        self.assertAlmostEqual(trained_network.biases[0][0], true_bias, None, None, 0.01)

    def test_delta_rule_with_learn_until_convergence(self):
        weights = [np.asarray([5])]
        biases = [np.asarray([6])]
        network = Network(weights, biases)
        training_data = [(0.1, 0.7026606543),
                         (0.2, 0.7150421057),
                         (0.3, 0.7271082163),
                         (0.4, 0.7388500061),
                         (0.5, 0.7502601056),
                         (0.6, 0.7613327148),
                         (0.7, 0.7720635494),
                         (0.8, 0.7824497764),
                         (0.9, 0.7924899414)]
        trained_network, iterations = learn_until_convergence(network, training_data, 0.00001, learning_rate=0.1)
        true_weight = 0.6
        true_bias = 0.8
        self.assertAlmostEqual(trained_network.weights[0][0], true_weight, None, None, 0.01)
        self.assertAlmostEqual(trained_network.biases[0][0], true_bias, None, None, 0.01)
