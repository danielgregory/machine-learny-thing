from random import shuffle

import numpy as np

from backmath import within_precision
from backpropagation import BackPropagation
from network import Network


def learn(network: Network, training_data, iterations, learning_rate=0.01):
    """
    Apply the training data to the weights and biases in the network.
    Return the updated network that has now been trained on the
    training_data. At each iteration we shuffle the training set to
    prevent getting stuck.
    """
    bp = BackPropagation(network)
    for _ in range(iterations):
        shuffle(training_data)
        _learn_once(bp, training_data, learning_rate)

    return network


def learn_until_convergence(network: Network, training_data, epsilon, learning_rate=0.01):
    """
    Keep iterating over the training data until convergence has been reached for
    both the biases and the weights.
    """

    bp = BackPropagation(network)
    iterations = 0
    while True:
        iterations += 1
        previous_weights = network.weights
        previous_biases = network.biases
        _learn_once(bp, training_data, learning_rate)
        if _network_has_converged(previous_weights, previous_biases, bp.network, epsilon):
            return bp.network, iterations

    raise Exception("Network does not converge to precision: " + epsilon)


def _learn_once(bp, training_data, learning_rate):
    """
    Complete one learning iteration over the training data. The
    network to be trained is a field on the BackPropagation object
    (bp) and is updated via side-effect.
    """
    for xs, target in training_data:
        bp.back_propagate(xs, target, learning_rate)


def learn_stuff(network: Network, training_data, iterations, learning_rate=0.01):
    for _ in range(iterations):
        shuffle(training_data)
        batched_training_data = batch_data(training_data, 3)

        # do all other batches
        for batch in batched_training_data:
            bp = BackPropagation(network)
            cumulative_cost_w, cumulative_cost_b = bp.back_propagate_version_2(*batch[0])
            for i, o in batch[1:]:
                cost_w, cost_b = bp.back_propagate_version_2(i, o)
                cumulative_cost_w = cumulative_cost_w + cost_w
                cumulative_cost_b = cumulative_cost_b + cost_b

            average_cost_w = cost_w / len(batch)
            average_cost_b = cost_b / len(batch)

            # update weights
            network.weights = [BackPropagation.gradient_descent(w, grad_w, learning_rate)
                               for w, grad_w in zip(network.weights, average_cost_w)]

            # update biases
            network.biases = [BackPropagation.gradient_descent(b, grad_b)
                              for b, grad_b in zip(network.biases, average_cost_b)]

    return network


def batch_data(training_data, batch_size):
    return [training_data[i:i + batch_size]
            for i in range(0, len(training_data), batch_size)]


def _network_has_converged(previous_weights, previous_biases, updated_network: Network, precision):
    """
    Compare the current network with the previously updated network. If the weights and
    biases are close enough to the previous weights and biases (where 'close enough' is
    defined by the value of epsilon) then we say the network has converged.
    """
    updated_weights = updated_network.weights
    updated_biases = updated_network.biases

    # compare weights
    for new_weight, old_weight in zip(updated_weights, previous_weights):
        for new_w, old_w in zip(np.nditer(new_weight), np.nditer(old_weight)):
            if not within_precision(old_w, new_w, precision):
                return False

    # compare biases
    for new_bias, old_bias in zip(updated_biases, previous_biases):
        for new_b, old_b in zip(np.nditer(new_bias), np.nditer(old_bias)):
            if not within_precision(old_b, new_b, precision):
                return False

    # if we've reached this point then all weights and biases must have converged to within
    # +/- epsilon, we therefore return true.
    return True
