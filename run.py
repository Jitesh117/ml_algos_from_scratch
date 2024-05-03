from knn import KNNClassifier
import numpy as np

# Dummy training data
X_train = np.array([[1, 1], [2, 2], [3, 1], [4, 4], [5, 3]])
y_train = np.array([0, 0, 1, 1, 1])

# Dummy test data
X_test = np.array([[2, 1], [4, 3]])

# Create and fit the KNN classifier
knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)

# Predict labels for the test data
y_pred = knn.predict(X_test)
print(y_pred)  # Output: [0. 1.]