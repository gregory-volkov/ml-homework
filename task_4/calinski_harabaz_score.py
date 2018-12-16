import numpy as np
from utils import normalize_labels


def calinski_harabaz_score(X, labels):
    labels = normalize_labels(labels)

    n_samples, _ = X.shape
    n_labels = len(set(labels))

    extra_disp, intra_disp = 0., 0.
    mean = np.mean(X, axis=0)
    for k in range(n_labels):
        cluster_k = np.array([
            X[i] for i in range(n_samples) if labels[i] == k
        ])
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)

    return (1. if intra_disp == 0. else
            extra_disp * (n_samples - n_labels) /
            (intra_disp * (n_labels - 1.)))