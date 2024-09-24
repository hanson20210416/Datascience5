import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    def __init__(self, alpha=0.01, lambda_=0.7, num_iter=1000):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.num_iter = num_iter
        self.theta = None
        self.cost_history = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        # Add bias term and scale features
        X = self.data_scaler(X)
        m, n = X.shape
        self.theta = np.zeros(n)
        self.cost_history = []

        # Perform gradient descent
        for _ in range(self.num_iter):
            y_pred = self._predict(X)
            cost = self.compute_cost_lin(X, y, self.theta)
            self.cost_history.append(cost)
            self.theta -= self.alpha * self.gradient_descent_lin(X, y, y_pred)
        return self

    def data_scaler(self, X):
        """Prepare data by scaling features and adding an intercept term."""
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))  
        return X_scaled

    def _predict(self, X):
        return X @ self.theta

    def compute_cost_lin(self, X, y, theta):
        """
        Compute cost for linear regression with regularization.
        """
        m = len(y)
        y_pred = X @ theta
        error = y_pred - y
        J = (1 / (2 * m)) * np.sum(error ** 2) + (self.lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
        return J

    def gradient_descent_lin(self, X, y, y_pred):
        """
        Perform gradient descent for linear regression.
        """
        m = len(y)
        error = y_pred - y
        gradient = (1 / m) * (X.T @ error)
        gradient[1:] += (self.lambda_ / m) * self.theta[1:]  
        return gradient

    def draw_costs(self):
        """Plot the cost history."""
        plt.plot(range(1, self.num_iter + 1), self.cost_history, color='b')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost (J)')
        plt.title('Cost function over iterations')
        plt.show()


class LogisticRegression:
    def __init__(self, alpha=0.01, lambda_=0.8, num_iter=1000):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.num_iter = num_iter
        self.theta = None
        self.cost_history = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        # Add bias term and scale features
        X = self.data_scaler(X)
        m, n = X.shape
        self.theta = np.zeros(n)
        self.cost_history = []

        # Perform gradient descent
        for _ in range(self.num_iter):
            y_pred = self._predict(X)
            cost = self.compute_cost_log(X, y, self.theta)
            self.cost_history.append(cost)
            self.theta -= self.alpha * self.gradient_descent_log(X, y, self.theta)
        return self

    def data_scaler(self, X):
        """Prepare data by scaling features and adding an intercept term."""
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))
        return X_scaled

    def _predict(self, X):
        return self.sigmoid(X @ self.theta)

    def compute_cost_log(self, X, y, theta):
        """
        Compute cost for logistic regression with regularization.
        """
        m = len(y)
        h = self.sigmoid(X @ theta)
        cost = -1 / m * (y @ np.log(h) + (1 - y) @ np.log(1 - h)) + (self.lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
        return cost

    def gradient_descent_log(self, X, y, theta):
        """
        Perform gradient descent for logistic regression.
        """
        m = len(y)
        h = self.sigmoid(X @ theta)
        gradient = (1 / m) * X.T @ (h - y)
        gradient[1:] += (self.lambda_ / m) * theta[1:] 
        return gradient

    def predict(self, X, threshold=0.5):
        """Predict binary labels (0 or 1) based on the logistic regression model."""
        X = self.data_preparation(X)
        probabilities = self._predict(X)
        return (probabilities >= threshold).astype(int)

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def draw_costs(self):
        """Plot the cost history."""
        plt.plot(range(1, self.num_iter + 1), self.cost_history, color='b')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost (J)')
        plt.title('Cost function over iterations')
        plt.show()


