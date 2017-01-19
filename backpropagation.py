import numpy as np

from backmath import sigmoid_derivative
from feedforward import FeedForward
from network import Network


class BackPropagation:
    def __init__(self, network: Network):
        self.network = network

    def back_propagate(self, xs, target, learning_rate=0.1):
        """
        Perform one backwards pass through the network, updating the state
        of the network as we go. i.e. update the network's weights and biases.
        """
        feed_forward = FeedForward(self.network)
        zs, activations = feed_forward.feed_forward(xs)
        deltas = self.deltas(zs, activations, target)
        cost_w = BackPropagation.cost_gradient_wrt_weight(activations, deltas)
        cost_b = BackPropagation.cost_gradient_wrt_bias(deltas)

        # update weights
        self.network.weights = [BackPropagation.gradient_descent(w, grad_w, learning_rate)
                                for w, grad_w in zip(self.network.weights, cost_w)]

        # update biases
        self.network.biases = [BackPropagation.gradient_descent(b, grad_b)
                               for b, grad_b in zip(self.network.biases, cost_b)]

    def back_propagate_version_2(self, xs, target, learning_rate=0.1):
        """
        Perform one backwards pass through the network, updating the state
        of the network as we go. i.e. update the network's weights and biases.
        """
        feed_forward = FeedForward(self.network)
        zs, activations = feed_forward.feed_forward(xs)
        deltas = self.deltas(zs, activations, target)

        cost_w = BackPropagation.cost_gradient_wrt_weight(activations, deltas)
        cost_b = BackPropagation.cost_gradient_wrt_bias(deltas)

        return np.asarray(cost_w), np.asarray(cost_b)

    def deltas(self, zs, activations, target):
        """
        Calculate all the deltas for each layer.
        """
        deltas = self.initialise_deltas(zs, activations, target)
        # start at the -2th layer as initialise_deltas
        # initialises outer layer delta.
        for i in range(2, len(zs) + 1):
            delta = self.next_delta(-i, zs, deltas)
            # use minus-indexing: we populate the array of deltas
            # backwards as we **back** propagate through the network.
            deltas[-i] = delta

        return deltas

    def initialise_deltas(self, zs, activations, target):
        """
        Initialise an array of length equal to the number of
        layers in the neural network (where the number of layers is
        given by len(zs)), then update the last element
         to equal the outer layer delta.
        """
        deltas = [0 for _ in range(len(zs))]
        outer_layer_delta = self.outer_layer_delta(zs, activations, target)
        deltas[-1] = outer_layer_delta
        return deltas

    def next_delta(self, layer, zs, deltas_so_far):
        """
        Propagate from the previous delta to the next inner delta. Remember
        we're back-propagating, so we're looping backwards over the network,
        and the concept of 'previous_layer' should be seen in this context.
        """
        previous_layer = layer + 1
        weight_transposed = self.network.weights[previous_layer].transpose()
        previous_deltas = deltas_so_far[previous_layer]
        # sigmoid_derivative evaluated at the current layer, z_l
        return np.dot(weight_transposed, previous_deltas) * sigmoid_derivative(zs[layer])

    # static helper methods below

    @staticmethod
    def outer_layer_delta(zs, activations, target):
        """
        Calculate delta for the final layer in the network, given by the
        element-wise product between the cost_derivative (which, for the outer
        layer, is just the difference between expected and actual) and
        the derivative of the sigmoid function, evaluated at the outer layer.
        """
        derivative_sigmoid = sigmoid_derivative(zs[-1])
        a = activations[-1]
        # perform element-wise multiplication on numpy arrays
        return BackPropagation.cost_derivative(target, a) * derivative_sigmoid

    @staticmethod
    def cost_gradient_wrt_weight(activations, deltas):
        """
        Get the partial derivative of the cost function with respect to
        the weight (== 'wrt' in function name) for each layer.
        """
        return [activation * delta for activation, delta in zip(activations[:-1], deltas)]

    @staticmethod
    def cost_gradient_wrt_bias(deltas):
        """
        Return the partial derivative of the cost function with respect to
        (== 'wrt' in function name) the bias for each layer.
        """
        return deltas

    @staticmethod
    def gradient_descent(x, grad_cost, learning_rate=0.1):
        """
        Update x to a new value, via gradient descent:
            x -> x' = x - rate * grad(cost)
        """
        return x - learning_rate * grad_cost

    @staticmethod
    def cost(target, actual):
        """
        Return the total cost function, representing a distance between
        the target and actual values. The factor of 0.5 is provided in
        order to make its derivative looker nicer.
        """
        return 0.5 * (target - actual) ** 2

    @staticmethod
    def cost_derivative(target, actual):
        """
        Return the derivative of the cost function.
        """
        return actual - target
