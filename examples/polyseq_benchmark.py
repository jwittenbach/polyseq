import polyseq as pseq
import timeit

def run_analysis():
    # load data
    data_dir = "/path/to/data"
    data = pseq.io.read_cellranger(data_dir)

    # remove rarely expressed genes
    dropped = data.drop_genes(num_cells=1)

    # normalize
    normed = dropped.log_normalize()

    # regress out total UMIs
    total = normed.sum(axis=1)
    regressed = pseq.regress(normed, total)

    # dimensionaity reduction
    reduced = pseq.dim.pca(regressed, k=15)

    # clustering
    clusters = pseq.clustering.graph_cluster(reduced, n_neighbors=15)
    reduced.clusters = clusters

    # tsne
    tsne = pseq.dim.multicore_tsne(reduced)

setup = "from __main__ import run_analysis"
cmd = "run_analysis()"
results = timeit.repeat(cmd, setup=setup, number=1, repeat=3)
print(results)
