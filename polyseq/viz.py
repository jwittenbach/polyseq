from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def single_violin(df, col, ax=None):
    sns.violinplot(y=df['group'], x=df[col], order=np.sort(df['group'].unique()),
                   orient="h", ax=ax, scale='width', bw=0.3, gridsize=50, inner=None, linewidth=2)
    ax.set_xlim([0, df.groupby('group')[col].apply(lambda x: np.percentile(x, 100)).max()])

def violin(data, groups, genes=None, figsize=(20, 20), clusterGenes=True):
    '''
    Draws a violin plot showing distributions of expression across groups

    Parameters:
    -----------
    data: DataFrame
        Pandas DataFrame of counts indexed as (cells, genes)
    groups: list
        List of group identifiers for cells of length data.shape[0]
    genes: list of strings, defualt=None
        List of genes to include in the plot. If None, then all genes will
        be used.
    figsize: tuple of lenth 2, default=(20, 20)
        Size of figure
    clusterGenes: bool, default=True
        Whether or not to do a hierarchical clusther on mean gene expreesion
        across groups in order to order genes.
    '''
    ncols = len(genes)
    if genes is None:
        subset = genes.copy()
    else:
        subset = data.copy()[genes]
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

def heatmap(data, figsize=(10, 10), cmap='viridis', row_names=False, col_names=True, col_rotation=30, log_norm=False, colorbar=False):
    '''
    Draws a heatmap of gene expression levels across cells

    Parameters:
    -----------
    data: DataFrame
        Pandas DataFrame of counts indexed as (cells, genes)
    figsize: tuple of lenth 2, default=(10, 10)
        Size of figure
    cmap: string containing valid PyPlot colormap, default='viridis'
        Color map to use
    rowNames: bool, default=True
        Include row labels
    colNames: bool, default=True
        Include column labels
    colNamesRot: numeric, default=0
        Angle by which to rotation column names (if present)
    logNorm: bool, default=False
        Apply log normalization to color map
    colorBar: bool, defualt=True
        Include a color bar.
    '''
    # aspect = h/w
    # figsize = (w, h)
    # data.shape = (h, w)
    if log_norm:
        from matplotlib.colors import LogNorm
        norm = LogNorm()
    else:
        norm = None
    sns.set_style('white')
    r = 1.0*figsize[1]/figsize[0]

    plt.figure(figsize=figsize)
    im = plt.imshow(data, cmap=cmap, aspect=r, interpolation='none', norm=norm)
    ax = plt.gca()

    if colorbar:
        if log_norm:
            maxDecade = np.floor(np.log10(data.max().max())).astype(int)
            ticks = [10**k for k in range(maxDecade+1)]
            plt.colorbar(im, ticks=ticks, fraction=0.045, pad=0.04)
        else:
            plt.colorbar(im, fraction=0.045, pad=0.04)
            #from matplotlib import ticker
            #cb = plt.colorbar(im, fraction=0.046, pad=0.04)
            #tick_locator = ticker.MaxNLocator(nbins=7)
            #cb.locator = tick_locator
            #cb.update_ticks()

    if row_names:
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels(data.index)

    if col_names:
        ax.set_xticks(range(data.shape[1]))
        ax.set_xticklabels(data.columns)
        ax.xaxis.tick_top()
        plt.xticks(rotation=col_rotation)
