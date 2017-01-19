import unittest

import numpy as np
import numpy.testing as npt

from feedforward import FeedForward
from backmath import inverse_sigmoid
from network import Network


class FeedForwardTest(unittest.TestCase):
    @staticmethod
    def test_two_one():
        """Very simple example of a feed forward with the inputs, biases
        and weights all set to 1. In this example, we imagine an input layer with
        2 nodes and an output layer with 1 node. There are no hidden layers."""
        weights = [np.asarray([[1, 1]])]
        biases = [np.asarray([1])]
        network = Network(weights, biases)
        feed_forwarder = FeedForward(network)

        # input to the neural network
        xs = np.asarray([1, 1])

        # unit under test
        _, outputs = feed_forwarder.feed_forward(xs)

        # assertions
        npt.assert_array_almost_equal([3.0], inverse_sigmoid(outputs[-1]))

    @staticmethod
    def test_three_two():
        """
        Three nodes in input layer, two nodes in output layer.
        """
        weights = [np.asarray([[1, 1, 1], [1, 1, 1]])]
        biases = [np.asarray([2, 3])]

        network = Network(weights, biases)
        feed_forwarder = FeedForward(network)

        # input to the neural network
        xs = np.asarray([1, 1, 1])

        # unit under test
        _, outputs = feed_forwarder.feed_forward(xs)

        # assertions
        npt.assert_array_almost_equal([5.0, 6.0], inverse_sigmoid(outputs[-1]))

    @staticmethod
    def test_three_two_one():
        """
        Test for a simple fully-connected graph: three inputs, two nodes
        in the hidden layer and two nodes in the output layer.
        """
        weights = [np.asarray([[1, 1, 1], [1, 1, 1]]),
                   np.asarray([[1, 1]])]
        biases = [np.asarray([1, 1]), np.asarray([1])]

        network = Network(weights, biases)
        feed_forwarder = FeedForward(network)

        # input to the neural network
        xs = np.asarray([1, 1, 1])

        # unit under test
        _, outputs = feed_forwarder.feed_forward(xs)
        # assertions
        npt.assert_array_almost_equal(np.asarray([0.950922]), outputs[len(outputs) - 1])

    @staticmethod
    def test_feed_forward_custom_activation_function_three_two_one():
        """
        Test feed forward algorithm with the identity function used as the
        activation function. Three neurons in first layer, two neurons in
        the second layer and one neuron in the final layer.
        """
        weights = [np.asarray([[1, 1, 1], [1, 1, 1]]),
                   np.asarray([[1, 1]])]
        biases = [np.asarray([1, 1]), np.asarray([1])]

        # input to the neural network
        xs = np.asarray([1, 1, 1])

        network = Network(weights, biases)
        # unit under test
        feed_forwarder = FeedForward(network)

        activation_function = lambda x: x

        _, outputs = feed_forwarder.feed_forward_custom_activation(xs, activation_function)
        npt.assert_array_almost_equal(np.asarray([9]), outputs[len(outputs) - 1])

    def test_feedforward_example_from_internet(self):
        """
        Network taken from here: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

        This test confirms that we get the same results.
        """
        weights = [np.asarray([[0.15, 0.25], [0.2, 0.3]]), np.asarray([[0.4, 0.5], [0.45, 0.55]])]
        biases = [np.asarray([0.35, 0.35]), np.asarray([0.6, 0.6])]
        network = Network(weights, biases)
        feed_forwarder = FeedForward(network)

        x = np.asarray([0.5, 0.1])
        zs, activations = feed_forwarder.feed_forward(x)

        self.assertAlmostEqual(0.76, activations[-1][0], None, None, 0.1)
        self.assertAlmostEqual(0.77, activations[-1][1], None, None, 0.1)
