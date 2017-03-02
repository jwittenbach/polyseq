from polyseq.clustering import _factor_error, _factor, nmf
import numpy as np

np.random.seed(0)

d1, d2, k = 400, 50, 3
w = np.random.uniform(size=(d1, k))
h = 10*np.random.uniform(size=(k, d2))
a = np.dot(w, h) + np.random.randn(d1, d2)
a[a<0] = 0

#w, h = _factor(a, k, 0, 0, 10)
#ahat = np.dot(w, h)

#print(a)
#print(ahat)
#print((a-ahat)**2)
#print(np.sum((a-ahat)**2))

#sse = _factor_error(a, k, 0, 0, 0.1, 10)
#print(a)

mse = nmf(a, 0.3, 6, [2, 3], [0, 0.001], [0, 0.001], nProcesses=4)
print(mse)
