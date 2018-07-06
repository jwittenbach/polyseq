import numpy as np
import pandas as pd

from polyseq.utils import cluster_arg_sort

class ExpressionMatrix(pd.DataFrame):

    def drop_cells(self, umis=None, num_genes=None, genes=None, umi_threshold=1):

        if isinstance(genes, (int, str)):
            genes = [genes]

        relevant = self if genes is None else self[genes]

        subset = self
        if umis is not None:
            subset = subset[relevant.sum(axis=1) >= umis]
        if num_genes is not None:
            subset = subset[(relevant >= umi_threshold).sum(axis=1) >= num_genes]

        return ExpressionMatrix(subset)

    def drop_genes(self, umis=None, num_cells=None, umi_threshold=1):

        subset = self
        if umis is not None:
            subset = subset.loc[:, subset.sum(axis=0) >= umis]
        if num_cells is not None:
            subset = subset.loc[:, (subset >= umi_threshold).sum(axis=0) >= num_cells]

        return ExpressionMatrix(subset)

    def downsample(self, fraction=None, number=None):

        if fraction is not None:
            if isinstance(fraction, (int, float)):
                fraction = (fraction, fraction)
            number = (int(np.round(f * s)) for f, s in zip(fraction, self.shape))

        if isinstance(number, int):
            number = (number, number)

        inds = [np.random.choice(np.arange(s), n, replace=False) for n, s in zip(number, self.shape)]

        return ExpressionMatrix(self.iloc[inds[0]].iloc[:, inds[1]])

    def sort(self, sort_cells=True, sort_genes=True, genes=None):
        if genes is None:
            genes = slice(None, None, None)
        elif isinstance(genes, (int, str)):
            genes = [genes]

        if sort_cells:
            sorted = self.loc[self[genes].sum(axis=1).sort_values(ascending=False).index]
        else:
            sorted = self

        if sort_genes:
            sorted = sorted[sorted.sum(axis=0).sort_values(ascending=False).index]

        return ExpressionMatrix(sorted)

    def cluster_sort(self, sort_cells=True, sort_genes=True):

        if sort_cells:
            inds = cluster_arg_sort(self)
            result = self.iloc[inds]
        else:
            result = self

        if sort_genes:
            inds = cluster_arg_sort(result.T)
            result = result.iloc[:, inds]

        return ExpressionMatrix(result)

    def log_normalize(self):
        return ExpressionMatrix(np.log(self + 1))

