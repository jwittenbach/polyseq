import numpy as np

def getLogW(data, Algorithm, k, algKwargs):
    '''
    Compute log(sum of squared distances from cluster)
    '''
    alg = Algorithm(n_clusters=k, **algKwargs)
    distsToCenters = alg.fit_transform(data)
    labels = alg.labels_
    logW = np.log(np.sum(np.choose(labels, distsToCenters.T)**2))
    return logW, labels 

def generateSample(bounds, mean, pca, n):
    '''
    Generate a sample from the null distribution
    '''
    proj = [np.random.uniform(low=l, high=h, size=n) for (l, h) in bounds]
    proj = np.array(proj).T
    orig = pca.inverse_transform(proj)
    return mean + orig

def gapStatistic(data, Algorithm, nSamples, algKwargs={}):
    '''
    Determine the number of clusters via the gap statistic method

    Tibsharani, Walther, & Hastie; J.R. Statist. Soc. B, 2001

    Parameters:
    -----------
    data: 2D array-like
        Data maxtrix of shape (samples, features)

    Algorithm: scikit-learn style clustering algorithm class
        e.g. sklearn.cluster.KMeans

    nSamples: int
        Number of samples from null distribution to use when computing
        reference statistics

    algKwargs: dictionary, default={}
       Optional keyword arguments to pass to the clustering algorithm
       constructor. 

    Returns:
    --------
    k: int
        The optimal number of clusters
    labels:
        Cluster labels when fitting with k clusters 
    '''
    n, k = data.shape

    # compute parameters of null distribution
    from sklearn.decomposition import PCA
    mean = data.mean(axis=0)
    centered = data - mean
    pca = PCA(n_components=k).fit(centered)
    proj = pca.transform(centered)

    bounds = [(v.min(), v.max()) for v in proj.T]
    k=1
    gapLast = 0
    labels = []
    while True:
        # cluster true data
        labelsLast = labels
        logW, labels = getLogW(data, Algorithm, k, algKwargs)
        # create and cluster data from null distribution
        stats = []
        #TODO: parallelize this loop for multithreaded execution
        for _ in range(nSamples):
            sample = generateSample(bounds, mean, pca, n)
            stat, _ = getLogW(sample, Algorithm, k, algKwargs)
            stats.append(stat)
        logWStar = np.mean(stats)
        error = np.sqrt(1-1.0/nSamples) * np.std(stats)
        gap = logWStar - logW
        deltaGap = gap - gapLast
        gapLast = gap
        k += 1
        if k == 2:
            continue
        if deltaGap < error:
            return k - 2, labelsLast
        if k > 10:
            print("exiting early")
            return -1, labels 

