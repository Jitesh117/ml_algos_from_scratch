import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fit the perceptron to the training data.

        Args:
            X (np.ndarray): Training data with shape (n_samples, n_features).
            y (np.ndarray): Target labels with shape (n_samples,).

        Returns:
            None
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.max_iterations):
            misclassified = 0

            for i in range(n_samples):
                prediction = self.predict(X[i])
                update = self.learning_rate * (y[i] - prediction)

                self.weights += update * X[i]
                self.bias += update

                if prediction != y[i]:
                    misclassified += 1

            if misclassified == 0:
                break

    def predict(self, X):
        """
        Predict the class label for a single sample.

        Args:
            X (np.ndarray): Sample with shape (n_features,).

        Returns:
            int: Predicted class label (0 or 1).
        """
        weighted_sum = np.dot(X, self.weights) + self.bias
        return int(weighted_sum >= 0)
