import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class CustomGaussianNB:
    def __init__(self):
        self.labels = None
        self.mu = None
        self.sd = None
        self.priors = None

    def fit(self, X_train, y_train):
        """Fit the model to training data"""
        self.labels = self.get_labels(y_train)
        self.priors = self.get_priors(y_train)
        self.mu, self.sd = self.get_likelihoods(X_train, y_train)
        return self

    def get_labels(self,y_train):
        """ Extract unique class labels from the training data """
        labels = np.unique(y_train)
        return labels
    
    def get_priors(self,y_train):
        """ Calculate prior probabilities for each unique label """
        priors = []
        total_samples = len(y_train)
        for label in self.labels:
            Pi = np.sum(y_train == label) / total_samples
            priors.append(Pi)
        return priors
    
    def get_likelihoods(self, X_train, y_train):
        """ Calculate mean and stdev matrices size (n_labels, n_features) """
        n_features = X_train.shape[1]
        n_labels = len(self.labels)
        mu = np.zeros((n_labels, n_features))
        sd = np.zeros((n_labels, n_features))
        for idx, label in enumerate(self.labels):
            for j in range(n_features):
                mu[idx][j] = np.mean(X_train[y_train == label, j])
                sd[idx][j] = np.std(X_train[y_train == label, j])
        return mu, sd  

    def predict_proba(self,X_test):
        '''Compute Gaussian Naive Bayes probabilities via likelihoods.'''
        Pyx = []
        for c in range(len(self.labels)):
            Pxy = np.exp(-0.5 * (X_test - self.mu[c]) ** 2 / (self.sd[c]**2)) / np.sqrt(2.0 * np.pi * (self.sd[c]**2))
            Py = self.priors[c]
            Pyx.append(Pxy.prod(axis=1) * Py)
        Pyx = np.array(Pyx).T
        return Pyx / Pyx.sum(axis=1, keepdims=True)
    
    def predict(self, X_test):
        """ Predict class labels for the test data """
        return np.argmax(self.predict_proba(X_test), axis=1)
    
if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    classifier = CustomGaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
