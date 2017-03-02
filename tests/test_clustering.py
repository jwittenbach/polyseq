import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polyseq.clustering import hCluster

d1 = 2/np.sqrt(2)
d2 = 0.5/np.sqrt(2)

s1 = 0.2
s2 = 0.2

data1 = s1*np.random.randn(100, 2)

data2_1 = d1*np.array([1, 1]) + d2*np.array([1, -1]) + s2*np.random.randn(5, 2)
data2_2 = d1*np.array([1, 1]) - d2*np.array([1, -1]) + s2*np.random.randn(50, 2)

data = pd.DataFrame(np.vstack([data1, data2_1, data2_2]))

#plt.plot(data[:, 0], data[:, 1], '.')
#plt.show()

from sklearn.cluster import KMeans
from polyseq.dimensionality import pca

res = hCluster(data, pca, KMeans, nSamples=100)
print(res)
