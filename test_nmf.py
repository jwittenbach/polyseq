from polyseq.clustering import _factor_error, _factor
import numpy as np

np.random.seed(0)

d1, d2, k = 10, 5, 2
w, h = np.abs(np.random.randn(d1, k)), np.abs(np.random.randn(k, d2))
a = np.dot(w, h) + 0.1*np.random.randn(d1, d2)

w, h = _factor(a, k, 0, 0, 10)
ahat = np.dot(w, h)

#print(a)
#print(ahat)
#print((a-ahat)**2)
#print(np.sum((a-ahat)**2))

sse = _factor_error(a, k, 0, 0, 0.1, 10)
print(a)
