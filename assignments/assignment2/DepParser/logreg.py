import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt

"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
Created 2019 by Dmytro Kalpakchi.
"""


class LogisticRegression(object):
    """
    This class performs logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    def __init__(self, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param theta    A ready-made model
        """

        if theta is not None:
            self.FEATURES = len(theta)
            self.theta = theta

        #  ------------- Hyperparameters ------------------ #
        self.LEARNING_RATE = 0.1  # The learning rate.
        self.MINIBATCH_SIZE = 256  # Minibatch size
        self.PATIENCE = 5  # A max number of consequent epochs with monotonously
        # increasing validation loss for declaring overfitting
        self.CONVERGENCE_MARGIN = 0.001
        # ---------------------------------------------------------------------- 

    def init_params(self, x, y):
        """
        Initializes the trainable parameters of the model and dataset-specific variables
        """
        # To limit the effects of randomness
        np.random.seed(524287)

        # Number of features
        self.FEATURES = len(x[0]) + 1

        # Number of classes
        self.CLASSES = len(np.unique(y))

        # Training data is stored in self.x (with a bias term) and self.y
        self.x, self.y, self.xv, self.yv = (
            self.train_validation_split(np.concatenate((np.ones((len(x), 1)), x), axis=1), y))

        # Number of datapoints.
        self.TRAINING_DATAPOINTS = len(self.x)

        # The weights we want to learn in the training phase.
        K = np.sqrt(1 / self.FEATURES)
        self.theta = np.random.uniform(-K, K, (self.FEATURES, self.CLASSES))

        # The current gradient.
        self.gradient = np.zeros((self.FEATURES, self.CLASSES))

        print("NUMBER OF DATAPOINTS: {}".format(self.TRAINING_DATAPOINTS))
        print("NUMBER OF CLASSES: {}".format(self.CLASSES))
        print("NUMBER OF FEATURES: {}".format(self.FEATURES))

    def train_validation_split(self, x, y, ratio=0.9):  # DONE
        """
        Splits the data into training and validation set, taking the `ratio` * 100 percent of the data for training
        and `1 - ratio` * 100 percent of the data for validation.

        @param x        A (N, D + 1) matrix containing training datapoints
        @param y        An array of length N containing labels for the datapoints
        @param ratio    Specifies how much of the given data should be used for training
        """

        # Shuffle data
        indices = np.random.permutation(len(x))
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        # Calculate the size of training set
        training_size = int(ratio * len(x))

        # Split the data
        xTraining, xValidation = x_shuffled[:training_size], x_shuffled[training_size:]
        yTraining, yValidation = y_shuffled[:training_size], y_shuffled[training_size:]

        return xTraining, yTraining, xValidation, yValidation

    def loss(self, x, y):
        """
        Calculates the loss for the datapoints present in `x` given the labels `y`.
        """

        loss = []  # loss vector for each pair (x[i], y[i])
        for i in range(len(x)):
            for k in range(self.CLASSES):
                if y[i] == k:
                    loss.append(-self.conditional_log_prob(y[i], x[i]))
                else:
                    loss.append(0)

        # Final cross-entropy is average over all elements in loss.
        return np.average(loss)

    def softmax(self, z):
        """
        The softmax function applied to a vector of values z = [z1, ..., zn].
        """
        result = []
        totalSum = 0
        for num in z:  # Calculate total sum for normalization
            totalSum += math.exp(num)
        for num in z:  # Calculate final vector
            result.append(math.exp(num) / totalSum)
        return result

    def conditional_log_prob(self, label, datapoint):
        """
        Computes the conditional log-probability log[P(label|datapoint)]
        """
        # theta is a matrix of size FEATURES x CLASSES
        # datapoint is a vector of size FEATURES (starts always with a 1)
        # label is a number and datapoint is a vector of features.
        # P(label | datapoint) = softmax(theta^T * datapoint)[label] In the og formula in slides, no need to transpose
        # theta because its dimensions are CLASSES x FEATURES, but we have it the other way around.
        return math.log(self.softmax(self.theta.T.dot(datapoint))[label])

    def compute_gradient(self, minibatch):  # DONE
        """
        Computes the gradient based on a mini-batch
        """

        # Gradient = (1 / numDatapoints) * X

        # Retrieving the datapoints of minibatch X and their corresponding label, minibatch
        # X: MINIBATCH_SIZE x FEATURES
        # Y: MINIBATCH_SIZE
        X, Y = self.x[minibatch], self.y[minibatch]

        # # Compute softmax(Xθ) ---------------------------------------------
        #
        # # XTheta: (MINIBATCH_SIZE x FEATURES) x (FEATURES x CLASSES) ==> MINIBATCH_SIZE x CLASSES
        # XTheta = X.dot(self.theta)
        # softXTheta = []  # Matrix of size MINIBATCH_SIZE x CLASSES containing all the applications of softmax
        # for datapoint in XTheta:
        #     softXTheta.append(self.softmax(datapoint))
        #
        # # -----------------------------------------------------------------
        #
        # # Compute gradient ------------------------------------------------
        # gradient = []  # Needs to be FEATURES x CLASSES, so we'll transpose it at the end.
        # for k in range(self.CLASSES):
        #     gradient_loss = np.zeros(self.FEATURES)
        #     for i in range(len(minibatch)):  # For each datapoint
        #         gradient_loss += X[i].dot(softXTheta[i][k] - (Y[i] == k))
        #     gradient.append(gradient_loss)
        # gradient = np.array(gradient).T / len(minibatch)
        # # -----------------------------------------------------------------
        # self.gradient = gradient

        oneHotY = np.zeros((Y.shape[0], self.CLASSES))  # One-hot label for Y: MINIBATCH_SIZE x CLASSES
        for i in range(self.CLASSES):
            oneHotY[:, i] = np.where(Y[:] == i, 1, 0)

        XTheta = X.dot(self.theta)  # XTheta: matrix of size MINIBTACH_SIZE x CLASSES
        softXTheta = []
        for row in XTheta:  # Apply softmax row by row to product X
            softXTheta.append(self.softmax(row))

        diff = softXTheta - oneHotY

        # Update the gradient
        self.gradient = X.T.dot(diff) / len(minibatch)

    def fit(self, x, y):  # DONE
        """
        Performs Mini-batch Gradient Descent.
        
        :param      x:      Training dataset (features)
        :param      y:      The list of training labels
        """
        self.init_params(x, y)
        self.init_plot(self.FEATURES)
        start = time.time()

        iteration, patience, prevLoss, converged = 0, 0, 0, False
        while not converged:
            iteration += 1

            datapoints = []
            for i in range(self.MINIBATCH_SIZE):  # Randomly pick MINIBATCH_SIZE datapoints
                random_datapoint = random.randrange(0, self.TRAINING_DATAPOINTS)
                datapoints.append(random_datapoint)

            # Compute gradient
            self.compute_gradient(datapoints)

            # Update matrix theta
            self.theta -= self.LEARNING_RATE * self.gradient

            # Compute cross entropy loss for the current iteration
            curLoss = self.loss(self.xv, self.yv)

            # Update patience
            patience = patience + 1 if curLoss > prevLoss else 0
            prevLoss = curLoss

            # Loss increases monotonously for PATIENCE measurements
            if patience > self.PATIENCE or curLoss < self.CONVERGENCE_MARGIN:
                converged = True

            if iteration == 1 or iteration % 100 == 0 or converged:
                print("Iter: {}, Cross-entropy loss: {} ".format(iteration, curLoss))
                self.update_plot(curLoss)

        print(f"Training finished in {time.time() - start} seconds")

    def get_log_probs(self, x):
        """
        Get the log-probabilities for all labels for the datapoint `x`
        :param      x:    a datapoint
        """
        if self.FEATURES - len(x) == 1:
            x = np.array(np.concatenate((np.array([1.]), x)))
        else:
            raise ValueError("Wrong number of features provided!")
        return [self.conditional_log_prob(c, x) for c in range(self.CLASSES)]

    def classify_datapoints(self, x, y):
        """
        Classifies datapoints
        """
        confusion = np.zeros((self.CLASSES, self.CLASSES))

        x = np.concatenate((np.ones((len(x), 1)), x), axis=1)

        num_datapoints = len(y)
        for d in range(num_datapoints):
            best_prob, best_class = -float('inf'), None
            for c in range(self.CLASSES):
                prob = self.conditional_log_prob(c, x[d])
                if prob > best_prob:
                    best_prob = prob
                    best_class = c
            confusion[best_class][y[d]] += 1

        self.print_result()
        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(self.CLASSES)))
        for i in range(self.CLASSES):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(self.CLASSES)))
        acc = sum([confusion[i][i] for i in range(self.CLASSES)]) / num_datapoints
        print("Accuracy: {0:.2f}%".format(acc * 100))

    def print_result(self):
        print("Theta: ")
        print(self.theta)
        print()
        print("Gradient: ")
        print(self.gradient)
        print()

    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
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
            self.lines[i], = (self.axes.plot([], self.val[0], '-',
                                             c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4))


def main():
    """
    Tests the code on a toy example.
    """

    def get_label(dp):
        if dp[0] == 1:
            return 2
        elif dp[2] == 1:
            return 1
        else:
            return 0

    from itertools import product
    x = np.array(list(product([0, 1], repeat=6)))

    #  Encoding of the correct classes for the training material
    y = np.array([get_label(dp) for dp in x])

    ind = np.arange(len(y))

    np.random.seed(524287)
    np.random.shuffle(ind)

    b = LogisticRegression()
    b.fit(x[ind][:-20], y[ind][:-20])  # Train on first 20 columns
    b.classify_datapoints(x[ind][-20:], y[ind][-20:])  # Test on last 20 columns


if __name__ == '__main__':
    main()
