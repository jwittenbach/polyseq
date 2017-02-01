import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def loadCellRanger(path, mapping=None):
    '''
    load cells-by-genes matrix from CellRanger data

    Parameters:
    -----------
    path: string
        path to directory of output "cellranger count"
    mapping: string, optional, default=None
        path to CSV file containing a mapping between gene IDs and gene names

    Returns:
    --------
    data: DataFrame
        Pandas DataFrame of unique transcript counts; rows = cells, cols = genes
    '''
    from scipy.io import mmread

    if path[-1] != '/':
        path = path + '/'

    data = mmread(path + 'matrix.mtx').toarray().T

    gene_ids = pd.read_csv(path + 'genes.tsv', sep='\t', header=None)

    if mapping is not None:
        mapping = pd.read_csv(mapping, header=None)
        genes = np.array(mapping.set_index(0)[1][gene_ids[0]])
    else:
        genes = gene_ids

    return pd.DataFrame(data, columns=genes)

def writeCSV(data, path, mapping=None):
    '''
    write data to an CSV file in (genes, cells) format

    Parameters:
    -----------
    data: DataFrame
        Pandas DataFrame of counts indexed as (cells, genes)
    path: string
        Path to output file
    mapping: string, default=None
        Path to a CSV file containing a mapping between gene IDs and gene
        names. If specified, gene IDs will be included in the file.
    '''
    final = data.T.reset_index().rename(columns={"index": "gene names"})
    if mapping is not None:
        mapping = pd.read_csv(mapping, header=None)
        final['gene ID'] = mapping.set_index(1)[0][data.columns].values
        d = final.shape[1]
        final = final.iloc[:, [d-1] + range(d-1)]

    final.to_csv(path, index=False)
