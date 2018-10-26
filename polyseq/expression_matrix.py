import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import coo_matrix

from polyseq.utils import cluster_arg_sort

class ExpressionMatrix(pd.DataFrame):

    def _finalize(self, index=None):
        if index is None:
            self.index = pd.Index.rename(self.index, "cell")
        else:
            self.index = index
        return self

    @property
    def clusters(self):
        if "cluster" in self.index.names:
            return np.array(self.index.get_level_values("cluster"))
    
    @clusters.setter
    def clusters(self, clusters):
        if not isinstance(self.index, pd.MultiIndex):
            self.index = pd.MultiIndex.from_arrays([self.index], names=self.index.names)

        if "cluster" in self.index.names:
            ind = self.index.names.index("cluster")
            new_labels = [
                labels if i != ind else clusters 
                for i, labels in enumerate(self.index.labels)
            ]
            new_levels = [
                levels if i != ind else np.unique(clusters)
                for i, levels in enumerate(self.index.levels)
            ]
            new_names = self.index.names
        else:
            new_labels = self.index.labels + [clusters]
            new_levels = self.index.levels + [np.unique(clusters)]
            new_names = self.index.names + ["cluster"]

        self.index = pd.MultiIndex(
            levels=new_levels, labels=new_labels, names=new_names)

    def get_cluster(self, i):
        return self.loc[self.clusters == i]

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
                fraction = (fraction, 1)
            number = (int(np.round(f * s)) for f, s in zip(fraction, self.shape))

        if isinstance(number, int):
            number = (number, self.shape[1])

        inds = [np.random.choice(np.arange(s), n, replace=False) if n < s
                else slice(None) for n, s in zip(number, self.shape)]

        return ExpressionMatrix(self.iloc[tuple(inds)])

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

    def to_cellranger(self, path):
        arr = coo_matrix(np.array(self).T)
        mmwrite(path + "matrix.mtx", arr)
        self.columns.to_series().to_csv(path + "genes.tsv", sep="\t")

    #def to_csv(self, path, mapping=None):
    #    '''
    #    write data to an CSV file in (genes, cells) format

    #    Parameters:
    #    -----------
    #    path: string
    #        Path to output file
    #    mapping: string, default=None
    #        Path to a CSV file containing a mapping between gene IDs and gene
    #        names. If specified, gene IDs will be included in the file.
    #    '''
    #    final = self.T.reset_index().rename(columns={"index": "gene names"})
    #    if mapping is not None:
    #        mapping = pd.read_csv(mapping, header=None)
    #        final['gene ID'] = mapping.set_index(1)[0][data.columns].values
    #        d = final.shape[1]
    #        final = final.iloc[:, [d-1] + range(d-1)]
    #
    #    final.to_csv(path, index=False)
