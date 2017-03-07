from polyseq.clustering import _factor_error, _factor, nmf
import numpy as np

np.random.seed(0)

d1, d2, k = 400, 50, 3
w = np.random.uniform(size=(d1, k))
h = 10*np.random.uniform(size=(k, d2))

w[np.random.uniform(size=w.shape) < 0.5] = 0
h[np.random.uniform(size=h.shape) < 0.5] = 0

a= np.dot(w, h) + 0.1*np.random.randn(d1, d2)
a[a<0] = 0

mse, vals = nmf(a, 0.01, 10, [2, 3, 4, 5], [0, 0.001, 0.01, 0.1], [0,
0.001, 0.01, 0.1], nProcesses=8)
