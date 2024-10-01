import numpy as np
from sklearn import metrics


def compute_ranked_lists(features, metric: str = "cosine") -> np.ndarray:
    ranked_lists = []

    if metric == "cosine":
        distances = metrics.pairwise.cosine_distances(features, features)
    else:
        distances = metrics.pairwise.euclidean_distances(features, features)
    for item in distances:
        rank_map = np.argsort(item)
        ranked_lists.append(rank_map)

    return np.asarray(ranked_lists)


def _computeAveragePrecision(rk, classes, n, d=1000) -> float:
    sumrj = 0
    curPrecision = 0
    sumPrecision = 0
    qClass = classes[rk[0]]
    for i in range(d):
        imgi = rk[i]
        imgiClass = classes[imgi]
        if qClass == imgiClass:
            sumrj = sumrj + 1
            posi = i + 1
            curPrecision = sumrj / posi
            sumPrecision += curPrecision
    class_size = int(n / (np.max(classes) + 1))
    nRel = class_size
    l = len(rk)
    avgPrecision = sumPrecision / min(l, nRel)
    return avgPrecision


def compute_map(ranked_lists, classes) -> float:
    acumAP = 0
    n = len(ranked_lists)
    for rk in ranked_lists:
        acumAP += _computeAveragePrecision(rk, classes, n)
    return acumAP / n
