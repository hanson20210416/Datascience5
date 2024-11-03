import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    def __init__(self, data, alpha=0.01, lambda_=0.1, num_iter=1000):
        self.data = data
        self.alpha = alpha
        self.lambda_ = lambda_
        self.num_iter = num_iter
        self.theta = None
        self.cost_history = None
        self.scaler = StandardScaler()

    def fit(self, X_scaled, y):
        # Add bias term and scale features
        m, n = X_scaled.shape
        self.theta = np.zeros(n)
        self.cost_history = []

        for _ in range(self.num_iter):
            y_pred = self._predict(X_scaled)
            cost = self.compute_cost(X_scaled, y, self.theta)
            self.cost_history.append(cost)
            self.theta -= self.alpha * self.gradient_descent(X_scaled, y, y_pred)
        return self

    def data_preparation(self):
        """Prepare data by scaling features and adding an intercept term."""
        X = self.data.iloc[:, :-1].values
        y = self.data.iloc[:, -1].values
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))  # Add intercept term
        return X_scaled, y

    def _predict(self, X_scaled):
        y_pred = X_scaled @ self.theta
        return y_pred

    def compute_cost(self, X_scaled, y, theta):
        """
        Compute cost for linear regression with regularization.
        """
        m = len(y)
        y_pred = X_scaled @ theta
        error = y_pred - y
        J = (1 / (2 * m)) * np.sum(error ** 2) + (self.lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
        return J

    def gradient_descent(self, X_scaled, y, y_pred):
        """
        Perform gradient descent for linear regression.
        """
        m = len(y)
        error = y_pred - y
        gradient = (1 / m) * (X_scaled.T @ error)
        gradient[1:] += (self.lambda_ / m) * self.theta[1:]  # Regularize
        return gradient

    def draw_costs(self):
        """Plot the cost history."""
        plt.plot(range(1, self.num_iter + 1), self.cost_history, color='b')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost (J)')
        plt.title('Cost function over iterations')
        plt.show()

def main():
    # Load data and create a model instance
    data = pd.read_csv('/Users/hezhipeng/Desktop/venv/DS/ML/delaney_solubility_with_descriptors.csv')
    model = LinearRegression(data)
    X_scaled, y = model.data_preparation()
    model.fit(X_scaled, y)
    model.draw_costs()
    #J =  model.compute_cost(X_scaled, y, model.theta)
    #print(J)

if __name__ == "__main__":
    main()