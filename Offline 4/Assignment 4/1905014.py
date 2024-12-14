import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.manifold import TSNE
from scipy.stats import poisson
import scipy.special
np.random.seed(14)

# data = np.loadtxt("pca_data.txt", delimiter=" ")
# print("Data shape:", data.shape)

def pca(data, top_n):
    mean =np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_centerd = (data - mean)/(std + 1e-8)

    cov_mat = np.cov(data_centerd, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov_mat)

    sort_index = np.argsort(eigval)[::-1] #reverse the ascending order
    eigval = eigval[sort_index]
    eigvec = eigvec[:, sort_index] #all row, sort columns

    selected = eigvec[:, :top_n] #all row, until top n columns

    transform = np.dot(data_centerd, selected)

    return transform

data1 = np.loadtxt("pca_data_online_2.txt", delimiter=" ")
data_projected = pca(data1, 2)
plt.scatter(data_projected[:,0], data_projected[:,1], s=5)
plt.title("PCA projection")
plt.show()

umap_model = umap.UMAP(n_components=2, random_state=42)
data_umap = umap_model.fit_transform(data1)
plt.scatter(data_umap[:, 0], data_umap[:, 1], s=5)
plt.title("UMAP Projection")
plt.show()
#data_umap[:, 0] = -data_umap[:, 0]

tsne_model = TSNE(n_components=2, random_state=42, perplexity=50)
data_tsne = tsne_model.fit_transform(data1)
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], s=5)
plt.title("t-SNE Projection")
plt.show()

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ax1.scatter(data_umap[:, 0], data_umap[:, 1], s=5)
# ax1.set_title("UMAP Projection", fontsize=15)

# ax2.scatter(data_tsne[:, 0], data_tsne[:, 1], s=5)
# ax2.set_title("t-SNE Projection", fontsize=15)

# plt.show()

class em_poisson:
    def __init__(self, k, max_iter=100, tol=1e-4):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def init_params(self, data):
        self.n = data.shape[0]
        self.lambdas = np.random.rand(self.k) * data.mean()
        self.pi = np.ones(self.k) / self.k

    def poisson_pmf(self, data, lamda):
        return ((lamda ** data) * np.exp(-lamda)) / scipy.special.factorial(data)
    
    def e_step(self, data):
        n_samples = data.shape[0]
        posteriors = np.zeros((n_samples, self.k))

        for i in range(self.k):
            posteriors[:, i] = self.pi[i] * self.poisson_pmf(data, self.lambdas[i])

        posteriors /= posteriors.sum(axis=1, keepdims=True)
        return posteriors
    
    def m_step(self, data, posteriors):
        for i in range(self.k):
            posterior = posteriors[:, i]
            total = posterior.sum()

            #self.pi[i] = total / data.shape[0] #np.mean(post??)
            self.pi[i] = np.mean(posterior)

            self.lambdas[i] = (data * posterior).sum() / total

    def find_em(self, data):
        self.init_params(data)
        log_likelihood = -np.inf
        for i in range(self.max_iter):
            posteriors = self.e_step(data)

            self.m_step(data, posteriors)

            new_log_likelihood = np.sum(np.log(np.sum([self.pi[i] * self.poisson_pmf(data, self.lambdas[i])
                                                       for i in range(self.k)], axis=0)))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                #print(i)
                break
            log_likelihood = new_log_likelihood
    
# X=np.array([1, 2])
# lam =2
# print(scipy.special.factorial(X))
# print(poisson.pmf(X, lam))
# print((lam ** X * np.exp(-lam)) / scipy.special.factorial(X))

data2 = np.loadtxt("em_data_online_2.txt")
#print(data2.shape)
em = em_poisson(2)
em.find_em(data2)

print("Estimated mean:", em.lambdas)

print("Proportions:", em.pi)
#print("Proportion of families without family planning:", em.pi[1])