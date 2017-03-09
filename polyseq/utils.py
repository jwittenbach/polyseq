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

def clusterArgSort(data):
    from sklearn.cluster import AgglomerativeClustering
    agg = AgglomerativeClustering()
    agg.fit(data)
    return expand_tree(agg.children_)

def parallelize(f, argList, nProcs):

    from multiprocessing import Process, Manager
    from numpy import asarray, argsort

    jobsPerProc = len(argList)//nProcs

    def g(f, args, pid, valStore):
        for i, a in enumerate(args):
            valStore[jobsPerProc*pid + i] = f(*a)

    with Manager() as manager:

        valStore = manager.dict()
        procs = []

        for i in range(nProcs):
            argSubset = argList[i*jobsPerProc : (i + 1)*jobsPerProc]
            p = Process(target=g, args=(f, argSubset, i, valStore))
            procs.append(p)

        for p in procs:
            p.start()

        for p in procs:
            p.join()

        for p in procs:
            p.terminate()

        keys, vals = valStore.keys(), valStore.values()
        return asarray(vals)[argsort(keys)]
