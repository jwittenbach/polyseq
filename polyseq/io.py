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
