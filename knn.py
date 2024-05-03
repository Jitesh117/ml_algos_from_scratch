import numpy as np
from collections import Counter

def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[j] for j in k_nearest_indices]
            y_pred[i] = Counter(k_nearest_labels).most_common(1)[0][0]
        
        return y_pred

        