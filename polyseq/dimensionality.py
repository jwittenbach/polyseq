import numpy as np
from .utils import parallelize
import warnings

def pca(data, k=None, nShuffles=1000, alpha=0.05, nProcesses=1):
    '''
    Principal components analysis with shuffle test
    '''

    from sklearn.decomposition import RandomizedPCA

    centered = data - data.mean(axis=0)
    zscored = centered/centered.std(axis=0)

    if k is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pca = RandomizedPCA(n_components=k)
        return pca.fit_transform(zscored)

    def bootstrapPC(seed):
        np.random.seed(seed)

        b = np.copy(zscored)
        nrows, ncols = b.shape
        for i in range(ncols):
            b[:, i] = b[:, i][np.random.permutation(nrows)]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pca = RandomizedPCA(n_components=1)
        pca.fit(b)
        return pca.explained_variance_[0]

    scores = parallelize(bootstrapPC, range(nShuffles), nProcesses)
    cutoff = np.percentile(scores, 100*(1-alpha))

    m = 100
    while True:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pca = RandomizedPCA(n_components=m)
        proj = pca.fit_transform(zscored)

        print('----')
        print(np.sort(scores))
        print(cutoff)
        print(pca.explained_variance_)

        try:
            ind = np.where(pca.explained_variance_ < cutoff)[0][0]
        except:
            m = 2*m
            print('looping...')
            continue
        print(ind)
        print(proj.shape)
        return proj[:, :ind]
