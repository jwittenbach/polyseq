import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors.kde import KernelDensity
from MulticoreTSNE import MulticoreTSNE
import umap as umap_module

from polyseq.utils import parallelize
from polyseq.expression_matrix import ExpressionMatrix


def tsne(data, **kwargs):
    '''
    wrapper for multicore TSNE algorithm

    note: multicore functionality does not work out-of-the-box of Mac OSX
    '''
    tsne = MulticoreTSNE(**kwargs).fit_transform(data)
    col_names = ["tsne-{}".format(i) for i in range(tsne.shape[1])]
    return ExpressionMatrix(tsne, columns=col_names)._finalize(index=data.index)

def umap(data, **kwargs):
    '''
    wrapper for umap algorithm
    '''
    embedding = umap_module.UMAP(**kwargs).fit_transform(data)
    col_names = ["umap-{}".format(i) for i in range(embedding.shape[1])]
    return ExpressionMatrix(embedding, columns=col_names)._finalize(index=data.index)


def pca(data, k=None, n_shuffles=100, alpha=0.05, n_processes=1, max_pcs=100, plot=False):
    '''
    Principal components analysis with shuffle test
    '''
    centered = data - data.mean(axis=0)
    zscored = centered/centered.std(axis=0)

    if k is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pca = RandomizedPCA(n_components=k)
        proj = pca.fit_transform(zscored)
        col_names = ["pc-{}".format(i) for i in range(proj.shape[1])]
        return ExpressionMatrix(proj, columns=col_names)._finalize(index=data.index)

    def bootstrap_pc(seed):
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

    args = [(i,) for i in range(n_shuffles)]
    scores = parallelize(bootstrap_pc, args, n_processes)
    cutoff = np.percentile(scores, 100 * (1 - alpha))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pca = RandomizedPCA(n_components=max_pcs)
    proj = pca.fit_transform(zscored)

    inds = np.where(pca.explained_variance_ < cutoff)[0]
    if inds.shape == (0,):
        print("all PCs computed are significant; you might try increasing max_pcs")
        n_pcs = max_pcs
    else:
        n_pcs = inds[0]
        proj = proj[:, :n_pcs]

    if plot:
        plt.figure(figsize=(20, 5))

        with plt.style.context(['seaborn-talk', 'seaborn-whitegrid']):

            # plot of distribution of bootstrapped PCs
            ax = plt.subplot(1, 2, 1)
            eps = 0.1
            min_score, max_score = (1 - eps) * scores.min(), (1 + eps) * scores.max()
            bandwidth = 3.0 / n_shuffles * (max_score - min_score)
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(scores[:, np.newaxis])
            s = np.linspace(min_score, max_score, 100)
            density = np.exp(kde.score_samples(s[:, np.newaxis]))
            ax.fill_between(s.squeeze(), 0, density.squeeze())
            ylim = [0, 1.1 * density.max()]
            plt.plot([cutoff, cutoff], ylim, '--r')
            plt.xlim(min_score, max_score)
            plt.ylim(ylim)
            plt.xlabel('bootstrapped variance explained')
            plt.ylabel('density')

            # plot of final PCs with cutoff
            plt.subplot(1, 2, 2)
            plt.plot(np.arange(max_pcs) + 1, pca.explained_variance_, 'o-')
            max_variance = pca.explained_variance_.max()
            xlim = [0, 1.1 * n_pcs]
            ylim = [0, 1.1 * max_variance]
            plt.plot(xlim, [cutoff, cutoff], '--r')
            plt.plot([n_pcs + 0.5, n_pcs + 0.5], ylim, '--g')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.xlabel('pc (ordered)')
            plt.ylabel('variance explained')

    col_names = ["pc-{}".format(i) for i in range(proj.shape[1])]
    return ExpressionMatrix(proj, columns=col_names)._finalize(index=data.index), scores, pca.explained_variance_
