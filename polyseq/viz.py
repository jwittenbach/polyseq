from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def single_violin(df, col, ax=None):
    sns.violinplot(y=df['group'], x=df[col], order=np.sort(df['group'].unique()),
                   orient="h", ax=ax, scale='width', bw=0.3, gridsize=50, inner=None, linewidth=2)
    ax.set_xlim([0, df.groupby('group')[col].apply(lambda x: np.percentile(x, 100)).max()])

def violin(data, groups, genes, figsize=(20, 20), clusterGenes=True):
    '''
    Makes a violin plot showing distributions of expression across groups

    Parameters:
    -----------
    data: DataFrame
        Pandas DataFrame of counts indexed as (cells, genes)
    groups: list
        List of group identifiers for cells of length data.shape[0]
    genes: list of strings
        List of genes to include in the plot
    figsize: tuple of lenth 2, default=(20, 20)
        Size of figure
    clusterGenes: bool, default=True
        Whether or not to do a hierarchical clusther on mean gene expreesion
        across groups in order to order genes.
    '''
    ncols = len(genes)
    subset = data[genes]
    subset['group'] = groups
        
    if clusterGenes:
        from .utils import clusterArgSort

        X = subset.groupby('group').mean().__array__().T
        order = clusterArgSort(X)
    else:
        order = range(len(genes))

    f, axes = plt.subplots(1, ncols, sharey=True, figsize=figsize)
    f.subplots_adjust(wspace=0)

    for i in range(len(genes)):
        ax, col = axes[i], subset.columns[order[i]] 
        ax.xaxis.set_visible(False)
        single_violin(subset, col, ax)
        ax.set_ylabel('')
        ax.set_title(col, rotation=45, y=1.08)
