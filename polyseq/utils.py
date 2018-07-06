import multiprocessing as mp

import numpy as np


def expand_tree(children):
    n = children.shape[0] + 1
    top = 2*n - 2

    def expand_node(k):
        if k < n:
            return [k]
        else:
            left, right = children[k - n]
            return expand_node(left) + expand_node(right)

    return expand_node(top)

def cluster_arg_sort(data):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph
    connectivity = kneighbors_graph(data, 50)
    agg = AgglomerativeClustering(linkage='ward', connectivity=connectivity)
    agg.fit(data)
    return expand_tree(agg.children_)


def run_function(f, job_args, process_id, n_processes, arr):
    for (i, args) in enumerate(job_args):
        result = f(*args)
        arr[process_id + i * n_processes] = result

def parallelize(func, args, n_processes):
    n_jobs = len(args)

    arr = mp.Array('d', np.zeros(n_jobs))
    procs = []

    for i in range(n_processes):
        job_args = args[i::n_processes]
        p = mp.Process(target=run_function, args=(func, job_args, i, n_processes, arr))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    return(np.array(arr[:]))
