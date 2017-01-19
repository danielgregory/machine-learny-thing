import numpy as np

from backmath import sigmoid
from network import Network


class FeedForward:
    def __init__(self, network: Network):
        self.network = network

    def feed_forward(self, x):
        """
        Feed forward. Return the activations at each layer
        using the default sigmoid activation function.

        """
        return self.feed_forward_custom_activation(x)

    def feed_forward_custom_activation(self, x, activation_f=lambda z: sigmoid(z)):
        """
        Feed forward algorithm with a custom activation function passed in
        as an argument (in the form of a lambda function, for example).
        The default activation function is the sigmoid function.
        """
        activations = [x]
        zs = []

        for w, b in zip(self.network.weights, self.network.biases):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            activations.append(activation_f(z))
        return zs, activations
