def f(x):
    return x * x


def f_primed(x):
    return 2 * x


def g(x):
    return (x + 3) ** 2


def g_primed(x):
    return 2 * (x + 3)


def gradient_descent(guess, rate=0.01, derivative=lambda x: f_primed(x)):
    updated_guess = guess - rate * derivative(guess)
    return updated_guess


def find_minimum_of_f(x):
    for i in range(1000):
        x = gradient_descent(x)
    return x


def find_minimum_of_g(x):
    for i in range(1000):
        x = gradient_descent(x, 0.01, g_primed)
    return x


def find_minimum_of_vector(v):
    return None