import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Center the data by subtracting the mean
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.X_centered = (X - self.mean)/(self.std+1e-8)

        # Compute the covariance matrix
        covariance_matrix = np.cov(self.X_centered, rowvar=False)

        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Sort the eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components eigenvectors (principal components)
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Project the data onto the principal components
        return np.dot(self.X_centered, self.components)

    def fit_transform(self, X):
        # Fit the PCA model and apply the transformation
        self.fit(X)
        return self.transform(X)


data = np.loadtxt("pca_data.txt", delimiter=" ")
pca = PCA(n_components=2)
X_projected = pca.fit_transform(data)

plt.scatter(X_projected[:,0], X_projected[:,1], s=5)
plt.show()