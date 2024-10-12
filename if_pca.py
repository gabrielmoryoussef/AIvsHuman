import numpy as np
from scipy import stats as scipy_stats
from scipy.linalg import eigh
from sklearn.cluster import KMeans

def ifpca(Data, K, KSvalue=None, pvalcut=None, rep=None, kmeansrep=30, per=1):
    p, n = Data.shape

    # Error checking
    if K is None:
        raise ValueError("Please include the number of clusters")

    if KSvalue is None:
        nullsimu = True
    else:
        nullsimu = False

    if pvalcut is None:
        pvalcut = np.log(p) / p

    if rep is None and nullsimu:
        rep = 100 * p

    # Normalize Data
    gm = np.mean(Data, axis=1)
    gsd = np.std(Data, axis=1)
    # Handle zero standard deviation
    gsd[gsd == 0] = 1
    Data = (Data - gm[:, np.newaxis]) / gsd[:, np.newaxis]

    # Simulate KS values
    if nullsimu:
        KSvalue = np.zeros(rep)
        kk = np.arange(n + 1) / n

        for i in range(rep):
            x = np.random.randn(n)
            z = (x - np.mean(x)) / np.std(x)
            z = z / np.sqrt(1 - 1/n)
            pi = scipy_stats.norm.cdf(z)
            pi.sort()

            KSvalue[i] = max(np.max(np.abs(kk[:n] - pi)), np.max(np.abs(kk[1:] - pi)))

        KSvalue *= np.sqrt(n)

    KSmean = np.mean(KSvalue)
    KSstd = np.std(KSvalue)
    if per < 1:
        KSvalue.sort()
        KSmean = np.mean(KSvalue[:round(rep*per)])
        KSstd = np.std(KSvalue[:round(rep*per)])

    # Calculate KS value for each feature in the data set
    kk = np.arange(n + 1) / n
    KS = np.zeros(p)
    for j in range(p):
        pi = scipy_stats.norm.cdf(Data[j] / np.sqrt(1 - 1/n))
        pi.sort()
        KS[j] = np.sqrt(n) * max(np.max(np.abs(kk[:n] - pi)), np.max(np.abs(kk[1:] - pi)))

    # Standardize KS value according to Efron's idea
    if per == 1:
        KS = (KS - np.mean(KS)) / np.std(KS) * KSstd + KSmean
    else:
        KS.sort()
        KSm = np.mean(KS[:round(per*p)])
        KSs = np.std(KS[:round(per*p)])
        KS = (KS - KSm) / KSs * KSstd + KSmean

    # Calculate p-value with simulated KS values
    pval = np.mean(KSvalue[:, np.newaxis] > KS, axis=0)
    ranking = np.argsort(pval)
    psort = pval[ranking]

    # Calculate HC functional at each data point
    kk = np.arange(1, p+1) / (1 + p)
    HCsort = np.sqrt(p) * (kk - psort) / np.sqrt(kk)
    HCsort = HCsort / np.sqrt(np.maximum(np.sqrt(n) * (kk - psort) / kk, 0) + 1)
    HC = np.zeros(p)
    HC[ranking] = HCsort

    # Decide the threshold
    Ind = np.argmax(psort > pvalcut)
    ratio = HCsort.copy()
    ratio[:Ind] = -np.inf
    ratio[round(p/2):] = -np.inf
    L = np.argmax(ratio[::-1])
    L = len(ratio) - L - 1
    numselect = L

    # Record the statistics for every feature
    stats_dict = {
        'KS': KS,
        'HC': HC,
        'pval': pval,
        'ranking': ranking
    }

    # IF-PCA
    data_select = Data[pval <= psort[L]]
    G = data_select.T @ data_select
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(G)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select the top K-1 eigenvectors
    V = eigenvectors[:, :K-1]

    kmeans = KMeans(n_clusters=K, n_init=kmeansrep, random_state=0)
    label = kmeans.fit_predict(V)

    return label, stats_dict, numselect