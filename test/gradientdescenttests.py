import unittest

from examples.gradientdescent import *


class TestGradientDescent(unittest.TestCase):
    def test_gradient_descent(self):
        guess = 5
        x_min = find_minimum_of_f(guess)
        self.assertAlmostEqual(0, x_min)

    def test_gradient_descent_of_g(self):
        guess = 5
        x_min = find_minimum_of_g(guess)
        self.assertAlmostEqual(-3, x_min)
