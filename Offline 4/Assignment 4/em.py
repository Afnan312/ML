import math
import numpy as np
from scipy.stats import poisson


class PoissonMixtureModelEM:
    def __init__(self, k, max_iter=100, tol=1e-4):
        self.k = k                  # Number of components
        self.max_iter = max_iter    # Maximum number of iterations
        self.tol = tol              # Convergence tolerance

    def initialize_params(self, X):
        n_samples = X.shape[0]
        
        # Initialize the rates (lambdas) for each component randomly
        self.lambdas = np.random.rand(self.k) * X.mean()
        
        # Initialize the mixing weights equally
        self.weights = np.ones(self.k) / self.k

    def poisson_pmf(self, X, lam):
        """Calculate the Poisson probability mass function."""
        #scipy.special.factorial(x)
        #return (lam ** X * np.exp(-lam)) / math.factorial(X)
        return poisson.pmf(X, lam)

    def e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.k))

        # Calculate the responsibilities (weights for each Poisson component)
        for i in range(self.k):
            responsibilities[:, i] = self.weights[i] * self.poisson_pmf(X, self.lambdas[i])
        
        # Normalize responsibilities across each sample
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        
        return responsibilities

    def m_step(self, X, responsibilities):
        # Update weights, means (lambdas)
        for i in range(self.k):
            responsibility = responsibilities[:, i]
            total_responsibility = responsibility.sum()

            # Update weights
            self.weights[i] = total_responsibility / X.shape[0] #np.mean(responsibility??)

            # Update lambda (mean of the Poisson distribution)
            self.lambdas[i] = (X * responsibility).sum() / total_responsibility

    def fit(self, X):
        self.initialize_params(X)
        
        log_likelihood = -np.inf
        for iteration in range(self.max_iter):
            # Expectation step
            responsibilities = self.e_step(X)

            # Maximization step
            self.m_step(X, responsibilities)

            # Calculate log-likelihood for convergence check
            new_log_likelihood = np.sum(np.log(np.sum([self.weights[i] * self.poisson_pmf(X, self.lambdas[i])
                                                       for i in range(self.k)], axis=0)))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

    #may not be needed

    def predict_proba(self, X):
        # Calculate the probabilities of each component for each sample
        responsibilities = self.e_step(X)
        return responsibilities

    def predict(self, X):
        # Assign each sample to the component with the highest probability
        responsibilities = self.predict_proba(X)
        return responsibilities.argmax(axis=1)

data = np.loadtxt("em_data.txt")
print(data.shape)
pmm = PoissonMixtureModelEM(k=2)
pmm.fit(data)

# Predict the cluster for each sample
predictions = pmm.predict(data)
# print("Cluster assignments:", predictions)

# Print the estimated lambda values (Poisson rates) for each component
print("Estimated Poisson rates (lambdas):", pmm.lambdas)

print("Proportion of families with family planning:", pmm.weights[0])
print("Proportion of families without family planning:", pmm.weights[1])