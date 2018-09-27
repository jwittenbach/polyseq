from pkg_resources import resource_filename
from subprocess import call

import pandas as pd
from scipy.io import mmread

from polyseq.expression_matrix import ExpressionMatrix

def read_mtx(exp_matrix_path, genes=None):
    arr = mmread(exp_matrix_path).toarray().T
    exp_matrix = pd.DataFrame(arr)
    if genes is not None:
        exp_matrix = exp_matrix.rename(genes, axis=1)
    return ExpressionMatrix(exp_matrix)._finalize()

def read_cellranger(path):
    genes = pd.read_csv(path + "/genes.tsv", delimiter='\t', header=None)[1]
    try:
        expression_matrix = read_mtx(path + "/matrix.mtx", genes=genes)
    except:
        expression_matrix = read_mtx(path + "/matrix.mtx.gz", genes=genes)
    return expression_matrix._finalize()

def read_pickle(path):
    return ExpressionMatrix(pd.read_pickle(path))

def load_example(example="brain"):
    '''
    options are "brain" and "vnc"
    '''
    path = resource_filename(__name__, "examples/sample_{}".format(example))
    return read_cellranger(path)

def download_example_data():
    from itertools import product

    path = resource_filename(__name__, "examples")
    url = "https://raw.githubusercontent.com/jwittenbach/polyseq/master/examples/"
    directories = [
        "/sample_brain",
        "/sample_vnc"
    ]
    file_names = [
        "/genes.tsv",
        "/matrix.mtx.gz"
    ]

    cmd = "cd {} && curl -O {}"
    for directory, file_name in product(directories, file_names):
        call(cmd.format(path + directory, url + directory + file_name), shell=True)

