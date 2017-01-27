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
