from pkg_resources import resource_filename

import pandas as pd
from scipy.io import mmread

from polyseq.expression_matrix import ExpressionMatrix

def read_mtx(exp_matrix_path, genes=None):
    arr = mmread(exp_matrix_path).toarray().T
    exp_matrix = pd.DataFrame(arr)
    if genes is not None:
        exp_matrix = exp_matrix.rename(genes, axis=1)
    return ExpressionMatrix(exp_matrix)

def read_cellranger(path):
    genes = pd.read_csv(path + "/genes.tsv", delimiter='\t', header=None)[1]
    try:
        expression_matrix = read_mtx(path + "/matrix.mtx", genes=genes)
    except:
        expression_matrix = read_mtx(path + "/matrix.mtx.gz", genes=genes)
    return expression_matrix

def read_pickle(path):
    return ExpressionMatrix(pd.read_pickle(path))

def load_example():
    path = resource_filename(__name__, "examples")
    return read_cellranger(path)
