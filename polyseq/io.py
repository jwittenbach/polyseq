from pkg_resources import resource_filename
from subprocess import call

import pandas as pd
from scipy.io import mmread
import tables

from polyseq.expression_matrix import ExpressionMatrix

def read_mtx(exp_matrix_path, genes=None):
    arr = mmread(exp_matrix_path).toarray().T
    exp_matrix = pd.DataFrame(arr)
    if genes is not None:
        exp_matrix = exp_matrix.rename(genes, axis=1)
    return ExpressionMatrix(exp_matrix)._finalize()

def read_cellranger(path):
    if path[-3:] == ".h5":
        expression_matrix = _load_cellranger_h5(path)
    else:
        expression_matrix = _load_cellranger_mtx(path)
    return expression_matrix

def _load_cellranger_h5(path):
    '''
    https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/advanced/h5_matrices
    '''
    with tables.open_file(path, 'r') as f:
        mat_group = list(f.iter_nodes(f.root))[0] # assuming first and only group is the correct one
        print("reading data")
        data = mat_group.data.read()
        print(data.dtype)
        print("reading indices")
        indices = mat_group.indices.read()
        print(indices.dtype)
        print("reading indptrs")
        indptr = mat_group.indptr.read()
        print(indptr.dtype)
        print("reading shape")
        shape = mat_group.shape.read()
        print(shape.dtype)
        print(shape)
        print("reading genes")
        genes = mat_group.gene_names.read()
    print("making sparse matrix")
    arr = csc_matrix((data, indices, indptr), shape=shape)#.toarray().T
    print("making dense matrix")
    return arr.toarray().T
    exp_matrix = pd.DataFrame(arr).rename(genes, axis=1)
    return ExpressionMatrix(exp_matrix)

def _load_cellranger_mtx(path):
    genes = pd.read_csv(path + "/genes.tsv", delimiter='\t', header=None)[1]
    try:
        expression_matrix = read_mtx(path + "/matrix.mtx", genes=genes)
    except:
        expression_matrix = read_mtx(path + "/matrix.mtx.gz", genes=genes)
    return expression_matrix

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

