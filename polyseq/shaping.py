def cull_cells(data, counts=1, genes=None, numGenes=None, geneCounts=1):
    '''
    remove all cells that do not meet some criteria for reads

    Parameters:
    -----------
    data: DataFrame
        Pandas DataFrame of counts indexed as (cells, genes)
    counts: int, default=1
        Number of total reads (summed across genes) above which a cell will
        be kept (inclusive). If None, this criterion will not be used.
    genes: list of strings, default=None
        A gene or list of genes to which to restrict the criteria.
    numGenes=None: int, default=None
        Number of genes that a cell expresses above which a cell will
        be kept (inclusive). If None, this criterion will not be used.
    geneCounts: int, default=1
        If numGenes is used, this is the number of reads above which a
        cell is considered to express a gene (inclusive).


    Returns:
    --------
    subset: DataFrame
        New DataFrame containing only cells that meet the criteria
    '''
    if counts is not None and numGenes is not None:
        print('Can only choose one criterion, either counts or numGenes')
        return

    if counts is None and numGenes is None:
        print('Must choose one criterion by setting either counts or numGenes')
        return

    if isinstance(genes, str):
        genes = [genes]

    if genes is not None:
        subset = data[genes]
    else:
        subset = data

    if counts is not None:
        crit = subset.sum(axis=1) >= counts
        subset = data[crit]

    if numGenes is not None:
        crit = (subset >= geneCounts).sum(axis=1) >= numGenes
        subset = data[crit]

    return subset

def cull_genes(data, counts=1, numCells=None, cellCounts=1):
    '''
    Remove all genes that do not meet a expression criterion, based on
    total counts and/or number of cells that express it.

    Default behavior with no optional parameters set is to drop all genes
    that are not expressed in any cell. Either counts or cellNum must
    not be None

    Parameters:
    -----------
    data: DataFrame
        Pandas DataFrame of counts indexed as (cells, genes)
    counts: int, default=1
        Number of total reads (summed across cells) above which a gene will
        be kept (inclusive). If None, this criterion will not be used
    numCells: int, default=None
        Number of cells in which a gene is expressed above which a gene will
        be kept (inclusive). If None, this criterion will not be used
    cellCounts: int, default=1
        If numCells is used, this the the number of reads above which
        a cell is considered to express this gene (inclusive)

    Returns:
    --------
    subset: DataFrame
        New DataFrame containing only the genes that meet the criteria
    '''

    if counts is None and numCells is None:
        print('Must choose at least one criterion by setting either counts or numCells')
        return

    if counts is not None:
        crit = data.sum() >= counts
        subset = data[crit.index[crit]]
    else:
        subset = data

    if numCells is not None:
        crit = (data >= cellCounts).sum() >= numCells
        subset = subset[crit.index[crit]]

    return subset

def sort(data, sortCells=True, sortGenes=True, genes=None):
    '''
    Sort cells and/or genes by total expression level

    Paramters:
    ----------
    data: DataFrame
        Pandas DataFrame of counts indexed as (cells, genes)
    sortCells: bool, default=True
        Whether or not to sort cells
    sortGenes: bool, default=True
        Whether or not to sort genes
    genes: list of strings, default=None
        If sorting cells, a gene or list of genes to use. If None, all genes
        will be used.

    Returns:
    --------
    sorted: DataFrame
        New DataFrame with cells and or rows sorted
    '''
    if sortCells:
        if genes is None:
            genes = slice(None, None, None)
        elif isinstance(genes, str):
            genes = [genes]
        result = data.loc[data[genes].sum(axis=1).argsort()[::-1]]
    else:
        result = data

    if sortGenes:
        result = result[result.sum(axis=0).sort_values()[::-1].index]

    return result

def cluster(data, sortCells=True, sortGenes=True):
    '''
    Sort cells and/or genes in an order given by clustering

    Useful for making heatmaps

    Parameters:
    -----------
    data: DataFrame
        Pandas DataFrame of counts indexed as (cells, genes)
    sortCells: bool, default=True
        Whether or not to sort cells
    sortGenes: bool, default=True
        Whether or not to sort genes

    Returns:
    --------
    sorted: DataFrame
        New DataFrame with cells and or rows sorted
    '''
    from .utils import clusterArgSort

    if sortCells:
        sortingInds = clusterArgSort(data.__array__())
        result = data.iloc[sortingInds]
    else:
        result = data

    if sortGenes:
        sortingInds = clusterArgSort(result.__array__().T)
        result = result.iloc[:, sortingInds]

    return result

