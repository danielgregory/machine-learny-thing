import unittest

import numpy as np
import numpy.testing as npt

from backmath import sigmoid, sigmoid_derivative
from backpropagation import BackPropagation
from feedforward import FeedForward
from network import Network


class BackPropagationTest(unittest.TestCase):
    @staticmethod
    def test_three_two_with_feed_forward():
        """
        More like an integration test. Combines feed forward with back propagation in the
        three-two neural network, as tested above.
        """
        target = np.asarray([2, 3])
        b = 1.0
        w = -b / 3
        x = np.asarray([1, 1, 1])
        weights = [np.asarray([[w, w, w], [w, w, w]])]
        biases = [np.asarray([b, b])]
        expected_outer_layer_delta = np.asarray([-0.375, -0.625])
        network = Network(weights, biases)

        # test feed forward with back propagation in concert
        ff = FeedForward(network)
        bp = BackPropagation(network)
        outer_layer_delta = bp.outer_layer_delta(*ff.feed_forward(x), target)

        # assertions
        npt.assert_array_almost_equal(expected_outer_layer_delta, outer_layer_delta)

    def test_initialise_deltas(self):
        """
        Assert that the length of the initialised delta array is equal to the
        number of layers in the network. i.e. the length of 'deltas' is equal
        to len(zs).
        """
        zs = [np.asarray([1, 2, 3]), np.asarray([1, 2, 3]), np.asarray([1, 2, 3])]

        # network, activations and target values not important for test.
        network = Network(None, None)
        activations = np.asarray([3])
        target = np.asarray([2])

        bp = BackPropagation(network)
        # unit under test
        deltas = bp.initialise_deltas(zs, activations, target)
        self.assertEqual(len(zs), len(deltas))

    @staticmethod
    def test_calculation_of_deltas_all_layers_in_two_two_one_net():
        """
        Calculate all the deltas and compare with values as calculated by hand.
        We're using a 2-2-1 network.
        """
        weights = [np.asarray([[1, 1, 1], [1, 1, 1]]),
                   np.asarray([[1, 1]])]
        biases = [np.asarray([1, 1]), np.asarray([1])]
        network = Network(weights, biases)
        ff = FeedForward(network)

        # input to the neural network
        x = np.asarray([1, 1, 1])
        zs, activations = ff.feed_forward(x)
        network = Network(weights, biases)
        bp = BackPropagation(network)

        # deltas = <class 'list'>: [array([-0.00086476, -0.00086476]), array([-0.04895949])]
        targets = np.asarray([2])
        deltas = bp.deltas(zs, activations, targets)

        npt.assert_almost_equal(-0.04895949, deltas[-1][0])
        npt.assert_almost_equal(-0.00086476, deltas[-2][0])
        npt.assert_almost_equal(-0.00086476, deltas[-2][1])

    @staticmethod
    def test_two_two_one_more_realistic():
        weights = [np.asarray([[1, 3], [4, 2]]),
                   np.asarray([[5, 6]])]
        biases = [np.asarray([1, 2]), np.asarray([3])]
        targets = np.asarray([2])
        network = Network(weights, biases)
        ff = FeedForward(network)

        # input to the neural network
        x = np.asarray([1, 2])
        zs, activations = ff.feed_forward(x)
        bp = BackPropagation(network)
        deltas = bp.deltas(zs, activations, targets)
        expected_deltas = BackPropagationTest._calculate_deltas_by_hand()
        npt.assert_almost_equal(expected_deltas[-1][0], deltas[-1][0])
        npt.assert_almost_equal(expected_deltas[-2][0], deltas[-2][0])

    @staticmethod
    def _calculate_deltas_by_hand():
        """
        'Private' helper method to calculate the deltas, as though
        doing so by hand. Much of this method's results are based
        on calculating the deltas manually.
        """
        z_1 = 8
        z_2 = 10
        z_3 = 5 * sigmoid(z_1) + 6 * sigmoid(z_2) + 3

        a_1 = sigmoid(z_1)
        a_2 = sigmoid(z_2)
        a_3 = sigmoid(z_3)

        target = 2
        sigmoid_primed = sigmoid_derivative(z_3)
        delta_outer_layer = (a_3 - target) * sigmoid_primed  # -8.42101453346e-07

        weights = np.asarray([5, 6]).transpose()
        sigmoid_primed2 = sigmoid_derivative(np.asarray([z_1, z_2]))
        delta_2 = np.dot(weights, delta_outer_layer) * sigmoid_primed2
        # delta_2 = [ -1.39651695e-09  -2.26929204e-10]
        return [delta_2, np.asarray([delta_outer_layer])]
