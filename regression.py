import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# sklearn imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, mean_squared_error, r2_score, confusion_matrix


class LinearRegression:
    def __init__(self, data, alpha=0.01, lambda_=0.1, num_iter=1000):
        self.data = data
        self.alpha = alpha
        self.lambda_ = lambda_
        self.num_iter = num_iter
        self.theta = None
        self.cost_history = None
        self.scaler = StandardScaler()

    def data_preparation(self,data):
        #X_1 = np.ones((data.shape[0],1))
        X_data = data.iloc[:,:-1].values
        #X = np.hstack((X_1,X_data))
        X = X_data
        y = data.iloc[:,-1]
        theta = np.zeros((X.shape[1],))
        return X, y, theta
    
    def gradient_descent(self,X, y, theta, alpha, num_iters):
        """
        prtfrom gradient descent
        """
        cost_history = np.zeros(num_iters)
        for i in range(num_iters):
            m = len(y)
            yhat = X @ theta
            error = yhat - y
            theta -= alpha / m * (X.T @ error)
            cost = np.mean(error**2)/2 
            cost_history[i] = cost       
        return theta, cost_history

    def reverse_theta(theta, scaler):
        """
        Parameters:
        theta (numpy array): Array of theta parameters obtained from the regression model.
        scaler (StandardScaler object): The scaler object used to scale the features.
        Returns:
        numpy array: transformed theta parameters in the context of the original, non-scaled features.
        """
        #initialize
        theta_original = np.zeros_like(theta)
        #tranform back intercept
        theta_original[0] = theta[0] - np.sum((theta[1:] * scaler.mean_) / scaler.scale_)
        #transform back coefficients
        theta_original[1:] = theta[1:] / scaler.scale_
        return theta_original
    
    def regularizatio(self, X, y, y_pred):
        m = X.shape[0]
        regularization = (self.lambda_ / (2 * m)) * np.sum(self.theta[1:]**2)
        return np.mean((y_pred - y)**2) / 2 + regularization
    
    def fit(self, X, y):
        # Add bias term and scale features
        X, y, theta = self.data_preparation(self,data)
        m, n = X.shape
        self.theta = np.zeros(n)
        # Perform gradient descent
        self.cost_history = []
        for _ in range(self.num_iter):
            y_pred = self._predict(X)
            cost = self._compute_cost(X, y, y_pred)
            self.cost_history.append(cost)
            self.theta -= self.alpha * self._compute_gradient(X, y, y_pred)
        return self
    
    def draw_costs(self): 
        """ function to draw historical cost"""
        plt.plot(range(1, self.num_iter + 1), self.cost_history, color='b')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost (J)')
        plt.title('Cost function over iterations')
        plt.show()


