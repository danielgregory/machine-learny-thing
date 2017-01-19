import unittest

import numpy.testing as npt
from backmath import *


class BackMathTest(unittest.TestCase):
    @staticmethod
    def test_sigmoid_with_np_array_input():
        expected = np.asarray([0.5, 0.5, 0.5])
        result = sigmoid(np.asarray([0, 0, 0]))
        npt.assert_array_equal(expected, result)

    def test_sigmoid_with_number_input(self):
        result = sigmoid(0)
        self.assertEqual(0.5, result)

    def test_inverse_sigmoid(self):
        result = inverse_sigmoid(0.5)
        self.assertEqual(0, result)

    def test_sigmoid_derivative(self):
        derivative = sigmoid_derivative(0)
        self.assertEqual(0.25, derivative)

    def test_is_not_within_epsilon(self):
        self.assertFalse(within_precision(3, 10, 1))

    def test_is_within_epsilon(self):
        self.assertTrue(within_precision(3, 3.5, 1))
