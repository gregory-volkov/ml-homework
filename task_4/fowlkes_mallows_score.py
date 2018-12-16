import numpy as np
from itertools import product
from utils import normalize_labels


def fowlkes_mallows_score(labels_true, labels_pred):
    n_samples = len(labels_true)

    def cont_matrix(a, b):
        a, b = normalize_labels(a), normalize_labels(b)
        len_a = len(set(a))
        len_b = len(set(b))
        m = np.empty((len_a, len_b), dtype=np.int)
        for i, j in product(range(len_a), range(len_b)):
            ids_a = np.where(a == i)
            ids_b = np.where(b == j)
            m[i, j] = len(np.intersect1d(ids_a, ids_b))

        return m

    c = cont_matrix(labels_true, labels_pred)
    tk = np.sum(np.multiply(c, c)) - n_samples
    pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
    qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples
    return np.sqrt(tk / pk) * np.sqrt(tk / qk) if tk != 0. else 0.
