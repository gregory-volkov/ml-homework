import numpy as np
from sklearn.metrics import pairwise_distances
from utils import normalize_labels


def davies_bouldin_score(X, labels):
    labels = normalize_labels(labels)
    dimension = len(X[0])
    n_labels = len(set(labels))
    n_samples, _ = X.shape
    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, dimension), dtype=np.float)

    for k in range(n_labels):
        cluster_k = np.array([
            X[i] for i in range(n_samples) if labels[i] == k
        ])

        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(
            pairwise_distances(
                cluster_k,
                [centroid]
            )
        )

    centroid_distances = pairwise_distances(centroids)
    score = (intra_dists[:, None] + intra_dists) / centroid_distances
    score[score == np.inf] = np.nan
    return np.mean(np.nanmax(score, axis=1))
