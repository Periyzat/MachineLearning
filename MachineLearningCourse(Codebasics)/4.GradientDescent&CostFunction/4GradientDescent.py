# importing libraries for numpy
import numpy as np


# defining gradient_decent function
def gradient_descent(x, y):
    # initialization
    m_curr = b_curr = 0  # initial guesses for the slope
    iterations = 1000  # number of iterations
    n = len(x)  # number of data points in dataset
    learning_rate = 0.001  # the step size for parameter updates in each iteration.
    # gradient decent iterations
    for i in range(iterations):
        # predicted values
        y_predicted = m_curr * x + b_curr
        # cost calculation - here it measures the error between the predicted values (y_predicted) and the actual target values (y).
        # The cost function is the mean squared error (MSE).
        cost = (1 / n) * sum([val ** 2 for val in (y - y_predicted)])
        # gradient computation - this calculates the gradients (partial derivatives) of the cost function with respect to model paramets "m" and "b".
        # this gradients are used to update the parameters in the direction that reduces the cost.
        md = -(2 / n) * sum(x * (y - y_predicted))
        bd = -(2 / n) * sum(y - y_predicted)
        # Parameter Updates - uodate the values of m_curr and b_curr using the gradients and the learning rate.
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost, i))


x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

gradient_descent(x, y)
