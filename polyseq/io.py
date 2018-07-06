import pandas as pd
from scipy.io import mmread

from polyseq.expression_matrix import ExpressionMatrix

def read_mtx(exp_matrix_path, genes=None):
    exp_matrix = pd.DataFrame(mmread(exp_matrix_path).toarray().T)
    if genes is not None:
        exp_matrix = exp_matrix.rename(genes, axis=1)
    return ExpressionMatrix(exp_matrix)

def read_cellranger(path):
    genes = pd.read_csv(path + "genes.tsv", delimiter='\t', header=None)[1]
    return read_mtx(path + "matrix.mtx", genes=genes)

def read_pickle(path):
    return ExpressionMatrix(pd.read_pickle(path))
