import polyseq as pseq
import numpy as np

np.random.seed(0)

lowdim = np.array([1, 0.5, 0.3])*np.random.randn(1000, 3)
p = np.random.randn(3, 10)
highdim = np.dot(lowdim, p)
data = highdim + 0.0*np.random.randn(*highdim.shape)

proj = pseq.dimensionality.pca(data, nShuffles=100, nProcesses=4)
print('----')
print(proj.shape)
