import numpy as np
import pandas as pd

class LogisticRegression:

    def __init__(self, learning_rate=0.01, num_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.weights = None

    def _sigmoid(self, x):
      return 1. / (1. + np.exp(-x))

    def fit(self, X, y):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing
                m binary 0.0/1.0 labels
        """
        X_ones = np.asarray(X)
        y_np = np.asarray(y).astype(float)
        X_squared = np.square(X_ones)
        X_new = np.hstack((X_ones, X_squared))

        # bias term to the input data
        X_ones = np.hstack((np.ones((X_new.shape[0], 1)), X_new))

        # init weights
        self.weights = np.zeros(X_ones.shape[1])

        # gradient descent
        for i in range(self.num_iterations):
            z = np.dot(X_ones, self.weights)
            predictions = self._sigmoid(z)
            error = predictions - y_np

            # update weights
            self.weights -= self.learning_rate * np.dot(X_ones.T, error)

#            # Printing cost every 100 iterations
#            if self.verbose and i % 100 == 0:
#                cost = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
#                print(f"Iteration: {i}, Cost: {cost}")

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        X_ones = np.asarray(X)
        X_squared = np.square(X_ones)
        X_new = np.hstack((X_ones, X_squared))

        # bias term to the input data
        X_ones = np.hstack((np.ones((X_new.shape[0], 1)), X_new))

        # calculating predictions
        z = np.dot(X_ones, self.weights)
        predictions = self._sigmoid(z)

        # probabilities to class labels
        class_labels = [1 if prob >= 0.5 else 0 for prob in predictions]

        return predictions

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        The average number of correct predictions
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true
    return correct_predictions.mean()


def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy

    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions

    Returns:
        Binary cross entropy averaged over the input elements
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise

    Hint: highly related to cross-entropy loss

    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.

    Returns:
        Element-wise sigmoid activations of the input
    """
    return 1. / (1. + np.exp(-x))
