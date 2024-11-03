import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem import Descriptors

class LogisticRegression:
    def __init__(self, data, alpha=0.01, lambda_=0.1, num_iter=1000):
        self.data = data
        self.alpha = alpha
        self.lambda_ = lambda_
        self.num_iter = num_iter
        self.theta = None
        self.cost_history = []
        self.scaler = StandardScaler()

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        # Perform gradient descent
        for _ in range(self.num_iter):
            y_pred = self._predict(X)
            cost = self.compute_cost_log(X, y, self.theta)
            self.cost_history.append(cost)
            self.theta -= self.alpha * self.gradient_descent_log(X, y, self.theta)
        return self

    def data_preparation(self):
        """Prepare data by scaling features and adding an intercept term."""
        X = self.data.iloc[:, :-1].values
        y = self.data.iloc[:, -1].values
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))  # Add intercept term
        return X_scaled, y

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
        gradient[1:] += (self.lambda_ / m) * theta[1:]  # Regularize
        return gradient

    def predict(self, X, threshold=0.5):
        """Predict binary labels (0 or 1) based on the logistic regression model."""
        X_scaled = self.scaler.transform(X)
        X_scaled = np.column_stack((np.ones(X_scaled.shape[0]), X_scaled))  # Add intercept term
        probabilities = self._predict(X_scaled)
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

def get_data():
    # Initialize the ChEMBL client
    chembl = new_client
    target_chembl_id = 'CHEMBL2034'  # Glucocorticoid receptor targets
    # Get bioactivity data for the target
    activity = chembl.activity.filter(target_chembl_id=target_chembl_id, standard_type='IC50')
    # Convert to DataFrame
    df_activity = pd.DataFrame(activity)
    # Select relevant columns and drop rows with missing values
    df_activity = df_activity[['molecule_chembl_id', 'canonical_smiles', 'standard_value']]
    df_activity.dropna(inplace=True)
    # Convert IC50 values to binary labels (active if IC50 < 1000 nM, inactive otherwise)
    df_activity['bioactivity_label'] = df_activity['standard_value'].apply(lambda x: 1 if float(x) < 1000 else 0)
    # Function to calculate molecular descriptors for smiles
    def calculate_descriptors(smiles):
        mol = Chem.MolFromSmiles(smiles)
        descriptors = {
            'MolecularWeight': Descriptors.MolWt(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol)
        }
        return pd.Series(descriptors)

    # Calculate descriptors for each molecule
    df_descriptors = df_activity['canonical_smiles'].apply(calculate_descriptors)
    df_GR = pd.concat([df_descriptors, df_activity['bioactivity_label']], axis=1)
    df_GR.dropna(inplace=True)  # Drop rows where descriptor calculation failed
    df_GR.to_csv('CHEMBL2034.csv', index=False)
    return df_GR

def main():
    data_df = get_data()
    model = LogisticRegression(data=data_df)
    X, y = model.data_preparation()
    model.fit(X, y)
    model.draw_costs()

if __name__ == "__main__":
    main()