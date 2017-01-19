import numpy as np


class Network:
    """
    Network class encapsulates the state of the network
    in the weights and biases. The weights and biases
    encode the entire structure of the network. Also included
    are some methods representing questions we can ask of the
    network.
    """

    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def len(self):
        """
        How many layers are there in the network?

        Count of the number of layers in the network. There
        will always be one more layer than there are lines
        (weights) between them.
        """
        return len(self.weights) + 1

    def weight_between(self, layer, this_node, that_node):
        """
        What is the weight between this node and that node?

        Tells us the weight between the first_node in the layer
        given by `layer`, and the second node in the next layer
        along.
        """
        weight = self.weights[layer]
        return weight[this_node, that_node]


def weights_shape(layers):
    """
    Given the list of layers, work out the shapes of the
    weight matrices (== numpy arrays).
    """
    tupled_layers = Network.as_tuples(layers)
    return [np.ndarray([columns, rows]) for (columns, rows) in tupled_layers]


def as_tuples(lst):
    """
    For example: [1,2,3,4,5] -> [(1,2), (2,3), (3,4), (4, 5)]
    """
    return [(lst[i], lst[i + 1]) for i in range(len(lst) - 1)]


def initialise_network(input_nodes_size, output_nodes_size, number_of_hidden_layers):
    """
    TODO what is the best way to initialise the weights and biases in
    a network?
    """
    weights = []
    biases = []
    return Network(weights, biases)
