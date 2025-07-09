from __future__ import print_function
import random
import numpy as np
import matplotlib.pyplot as plt

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""


class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.01  # The learning rate.
    CONVERGENCE_MARGIN = 0.001  # The convergence criterion.
    MAX_ITERATIONS = 50000  # Maximal number of passes through the datapoints in gradient descent.
    MINIBATCH_SIZE = 1000  # Minibatch size (only for minibatch gradient descent)
    WEIGHTED_LOSS = True  # Use weighted loss?

    # ----------------------------------------------------------------------

    def __init__(self, x=None, y=None, Nn=0, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINTS*FEATURES array.
        @param y The labels as a DATAPOINTS array.
        @param theta A ready-made model. (instead of x and y)
        """

        self.i = None
        self.val = None
        self.axes = None
        self.lines = None

        # np.vectorize can be used so that we can use our sigmoid function on vectors
        self.sigmoid_v = np.vectorize(self.sigmoid)

        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:

            # x is a matrix of size DATAPOINTS x FEATURES

            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)

            # wn and wp parameters for weighted loss
            self.wp = Nn / self.DATAPOINTS if self.WEIGHTED_LOSS else 1
            self.wn = 1 - self.wp if self.WEIGHTED_LOSS else 1

    # ----------------------------------------------------------------------

    def displayInfo(self, iteration, termination=False):
        # Plot only every 100 iterations.
        if iteration == 1 or iteration % 100 == 0 or termination:
            if not termination:
                print("Iter: {}, sum of squares: {} ".format(iteration, np.sum(np.square(self.gradient))))
            else:
                print("Converged: Iter {}, with sum of squares: {}:".format(iteration, np.sum(
                    np.square(self.gradient))))
            self.update_plot(np.sum(np.square(self.gradient)))

    def checkConvergence(self):
        return np.sum(np.square(self.gradient)) < self.CONVERGENCE_MARGIN

    def selectDataPoints(self):
        datapoints = []
        for i in range(self.MINIBATCH_SIZE):
            random_datapoint = random.randrange(0, self.DATAPOINTS)
            datapoints.append(random_datapoint)
        return datapoints

    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / (1 + np.exp(-z))

    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """
        sigma = self.sigmoid(self.theta.dot(self.x[datapoint].T))
        return sigma if label == 1 else 1 - sigma

    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """

        wp = self.wp
        wn = self.wn
        y = self.y

        # for k in range(len(self.gradient)):
        #     sum = 0
        #     for i in range(self.DATAPOINTS):
        #         x_i = self.x[i]  # x_i = x(i)_1, ..., x(i)_n,
        #         x_i_theta = x_i.dot(self.theta)
        #         sum += x_i[k] * (self.sigmoid(-x_i_theta) * (wn * y[i] - wp * y[i] - wn) + wn * (1 - y[i]))
        #     self.gradient[k] = sum
        # self.gradient = self.gradient / self.DATAPOINTS

        # Without loops
        xTheta = self.x.dot(self.theta)
        self.gradient = self.x.T.dot(self.sigmoid_v(-xTheta) * (wn * y - wp * y - wn) + wn * (1 - y))
        self.gradient = self.gradient / self.DATAPOINTS

    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a minibatch
        (used for minibatch gradient descent).
        """

        # Get corresponding matrices X of datapoints and Y of labels
        X, Y = self.x[minibatch], self.y[minibatch]

        wp = self.wp
        wn = self.wn

        # for k in range(len(self.gradient)):
        #     sum = 0
        #     for i in range(len(X)):
        #         x_i_theta = X[i].dot(self.theta)
        #         sum += X[i][k] * (self.sigmoid(-x_i_theta) * (wn * Y[i] - wp * Y[i] - wn) + wn * (1 - Y[i]))
        #     self.gradient[k] = sum
        # self.gradient = self.gradient / len(X)

        # Without loops
        xTheta = X.dot(self.theta)
        self.gradient = X.T.dot(self.sigmoid_v(-xTheta) * (wn * Y - wp * Y - wn) + wn * (1 - Y))
        self.gradient = self.gradient / len(minibatch)

    def compute_gradient(self, datapoint):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """

        wp = self.wp
        wn = self.wn
        x = self.x[datapoint]
        y = self.y[datapoint]

        # for k in range(len(self.gradient)):
        #     x_i_theta = x.dot(self.theta)
        #     self.gradient[k] = x[k] * (self.sigmoid(-x_i_theta) * (wn * y - wp * y - wn) + wn * (1 - y))

        xTheta = x.dot(self.theta)
        self.gradient = x.dot(self.sigmoid(-xTheta) * (wn * y - wp * y - wn) + wn * (1 - y))

    def stochastic_fit(self):
        """
        Performs Stochastic Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        # Initialise iteration to 0. Will be used for tracking
        iteration = 0
        converged = False
        while (not converged) and (iteration < self.MAX_ITERATIONS):
            iteration += 1
            index = random.randrange(0, self.DATAPOINTS)  # Get random i in interval [0, DATAPOINTS)
            self.compute_gradient(index)

            # Update theta
            self.theta -= self.LEARNING_RATE * self.gradient

            self.displayInfo(iteration)
            converged = self.checkConvergence()

        self.displayInfo(iteration, converged)

    def minibatch_fit(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        # Initialise iteration to 0. Will be used for tracking
        iteration = 0
        converged = False
        while (not converged) and (iteration < self.MAX_ITERATIONS):
            iteration += 1
            datapoints = self.selectDataPoints()  # Pick MINIMBATCH_SIZE datapoints randomly
            self.compute_gradient_minibatch(datapoints)

            self.theta -= self.LEARNING_RATE * self.gradient

            self.displayInfo(iteration)
            converged = self.checkConvergence()

        self.displayInfo(iteration, converged)

    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)

        iteration = 0  # Used for formatting
        converged = False
        while (not converged) and (iteration < self.MAX_ITERATIONS):
            iteration += 1

            self.compute_gradient_for_all()
            self.theta -= self.LEARNING_RATE * self.gradient

            self.displayInfo(iteration)
            converged = self.checkConvergence()

        self.displayInfo(iteration, converged)

    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        # print('Model parameters:')
        # print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))
        accuracy = ((confusion[0][0] + confusion[1][1]) /
                    (confusion[0][0] + confusion[1][1] + confusion[0][1] + confusion[1][0]))
        print("Accurary: {:.2f}% ".format(accuracy*100))

    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))

    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if not self.i:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)

    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines = []

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5,
                                            markersize=4)

    # ----------------------------------------------------------------------
