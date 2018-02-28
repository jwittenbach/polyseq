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
    agg = AgglomerativeClustering()
    agg.fit(data)
    return expand_tree(agg.children_)

def parallelize(f, arg_list, n_processes):

    from multiprocessing import Process, Manager
    from numpy import asarray, argsort

    jobs_per_proc = len(arg_list) // n_processes

    def run_function(f, args, pid, val_store):
        for i, a in enumerate(args):
            val_store[jobs_per_proc * pid + i] = f(*a)

    with Manager() as manager:

        val_store = manager.dict()
        procs = []

        for i in range(n_processes):
            args = arg_list[i * jobs_per_proc: (i + 1) * jobs_per_proc]
            p = Process(target=run_function, args=(f, args, i, val_store))
            procs.append(p)

        for p in procs:
            p.start()

        for p in procs:
            p.join()

        for p in procs:
            p.terminate()

        keys, vals = val_store.keys(), val_store.values()
        return asarray(vals)[argsort(keys)]
