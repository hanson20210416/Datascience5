import numpy as np
import pandas as pd
from collections import Counter
from logger import *
import custom_plotter as plotter

class ClassifierID3:
    def __init__(self, max_depth=None, criterion='entropy'):
        self.max_depth = max_depth
        self.criterion = criterion

    # Static method for entropy calculation
    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return -np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-100)))

    # Information gain method, which uses the entropy method
    def information_gain(self, data, feature, target):
        initial_entropy = self.entropy(data[target])
        values, counts = np.unique(data[feature], return_counts=True)
        weighted_entropy = sum((counts[i] / sum(counts)) * self.entropy(data[data[feature] == v][target]) 
                               for i, v in enumerate(values))
        return initial_entropy - weighted_entropy

    def fit(self, X_train, y_train):
        classes = np.unique(y_train)
        y_onehot = (y_train[:, np.newaxis] == classes).astype(float)
        self.best_feature, self.best_threshold = self.determine_split(X_train, y_onehot)
        self.classes_ = classes
        return self

    def determine_split(self, X_train, y_onehot):
        cost_function = self.__class__.criteria[self.criterion]
        best_feature = best_threshold = None
        best_cost = cost_function(y_onehot.mean(axis=0))
        for feature in range(X_train.shape[1]):
            ordered = np.unique(X_train[:, feature])
            for threshold in (ordered[:-1] + ordered[1:]) / 2.0:
                subset = (X_train[:, feature] <= threshold)
                fraction = subset.mean()
                cost = (fraction * cost_function(y_onehot[subset, :].mean(axis=0)) +
                        (1.0 - fraction) * cost_function(y_onehot[~subset, :].mean(axis=0)))
                if cost < best_cost:
                    best_feature, best_threshold, best_cost = feature, threshold, cost
        return best_feature, best_threshold

    def predict_proba(self, X_train, y_train, X_test):
        classes = np.unique(y_train)
        y_onehot = (y_train[:, np.newaxis] == classes).astype(float)
        feature, threshold = self.determine_split(X_train, y_onehot)
        subset_test = (X_test[:, feature] <= threshold)
        
        y_proba = np.empty((X_test.shape[0], len(classes)))
        y_proba[subset_test, :] = y_onehot[X_train[:, feature] <= threshold].mean(axis=0)
        y_proba[~subset_test, :] = y_onehot[X_train[:, feature] > threshold].mean(axis=0)
        
        return y_proba

    def predict(self, X_train, y_train, X_test):
        y_proba = self.predict_proba(X_train, y_train, X_test)
        return np.argmax(y_proba, axis=1)

    def id3(self, data, features, target, depth=0):
        if len(set(data[target])) == 1:
            return data[target].iloc[0]
        
        if self.max_depth is not None and depth >= self.max_depth:
            return data[target].mode()[0]

        gains = {feature: self.information_gain(data, feature, target) for feature in features}
        best_feature = max(gains, key=gains.get)

        if gains[best_feature] < 1e-5:
            return data[target].mode()[0]

        tree = {best_feature: {}}
        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]
            remaining_features = [f for f in features if f != best_feature]
            subtree = self.id3(subset, remaining_features, target, depth + 1)
            tree[best_feature][value] = subtree
        return tree

if __name__ == "__main__":
    data = pd.read_csv('/Users/hezhipeng/Desktop/venv/DS/ML/sample.txt')
    target = 'target_column'
    features = list(data.columns.difference([target]))

    classifier = ClassifierID3(max_depth=2)
    decision_tree = classifier.id3(data, features, target)
    print("Decision Tree:", decision_tree)
