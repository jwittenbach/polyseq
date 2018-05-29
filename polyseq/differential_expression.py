import numpy as np
from sklearn.svm import LinearSVC

def upregulated(data, clusters, n=20):
    '''
    computes top features that differentiate each cluster from others using SVM

    Parameters
    ----------
    data: ndarray
        2D array that is (observations, features)
    clusters: ndarray
        1D array of ints representing cluster lables of lenth (observations)
    n: int
        Number of top features to compute

    Returns
    -------
    features: ndarray
        2D array containing the sorted indices the top features defining each
        cluster, or size (clusters, n)
    '''

    data = data.__array__()

    svc = LinearSVC()
    svc.fit(data, clusters)

    results = []

    for w in svc.coef_:
        z = w / np.sqrt((w**2).sum())
        pos_inds = np.where(z > 0)[0]
        pos_vals = z[pos_inds]
        pos_top_k = np.argsort(-pos_vals)[:n]
        top_k = pos_inds[pos_top_k]
        results.append(top_k)

    return np.array(results)
