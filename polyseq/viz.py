import collections

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.neighbors.kde import KernelDensity


STYLE_CONTEXTS = ['seaborn-talk', 'seaborn-whitegrid']


def violins(data, genes, group_by=None, cluster_genes=True, figsize=(20, 20)):
    ncols = len(genes)
    subset = data.copy()[genes]

    groups = data.index.get_level_values(group_by)

    if groups is not None:
        subset['group'] = groups

        if cluster_genes:
            from .utils import cluster_arg_sort

            X = subset.groupby('group').mean().__array__().T
            order = cluster_arg_sort(X)

    if not cluster_genes:
        order = range(len(genes))

    f, axes = plt.subplots(1, ncols, sharey=True, figsize=figsize)
    f.subplots_adjust(wspace=0)

    cmap = mpl.cm.get_cmap('tab10')

    for i in range(len(genes)):
        c = cmap((i % 10) / 10)
        ax, col = axes[i], subset.columns[order[i]] 
        ax.xaxis.set_visible(False)
        sns.violinplot(y=groups, x=data[col], order=np.sort(np.unique(groups)), color=c,
                       orient="h", ax=ax, scale='width', bw=0.3, gridsize=50, inner=None, linewidth=2)
        ax.set_ylabel('')
        ax.set_title(col, rotation=45, y=1.08)
        if groups is not None:
            ax.set_xlim([0, subset.groupby('group')[col].apply(lambda x: np.percentile(x, 100)).max()])

def scatter(data, color_by=None, cmap=None, **kwargs):
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]

    if color_by is None:
        plt.scatter(x, y, cmap=cmap, **kwargs)
    elif isinstance(color_by, collections.Iterable) and not isinstance(color_by, str):
        cmap = 'Blues' if cmap is None else cmap
        plt.scatter(x, y, c=color_by, cmap=cmap, **kwargs)
    elif color_by in data.index.names:
        color_idx = data.index.names.index(color_by)
        color_levels = data.index.levels[color_idx]
        color_labels = data.index.labels[color_idx]

        n_levels = len(color_levels)
        cmap = 'gist_stern' if cmap is None else cmap
        cmap = mpl.cm.get_cmap(cmap)
        if isinstance(cmap, mpl.colors.ListedColormap):
            colors = np.array(cmap.colors)[np.arange(n_levels) % len(cmap.colors)]
        else:
            colors = cmap(np.arange(n_levels, dtype='float') / n_levels)[:, :-1]

        for i in range(n_levels):
            mask = color_labels == i
            label = "{} {}".format(color_by, i)
            plt.scatter(x[mask], y[mask], c=[colors[i]], label=label, **kwargs)

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

def kde_plot(samples, ax=None, eps=0.1, bw_factor=10.0):
    if ax is None:
        ax = plt.gca()

    n_samples = len(samples)

    min_score, max_score = (1 - eps) * samples.min(), (1 + eps) * samples.max()
    bandwidth = bw_factor * (max_score - min_score) / n_samples

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(samples[:, np.newaxis])
    s = np.linspace(min_score, max_score, 100)
    density = np.exp(kde.score_samples(s[:, np.newaxis]))

    ax.fill_between(s.squeeze(), 0, density.squeeze())
    ylim = [0, (1 + eps) * density.max()]
    ax.set_xlim(min_score, max_score)
    ax.set_ylim(*ylim)


