from itertools import combinations


def rand_score(labels_true, labels_pred):
    n_samples = len(labels_true)
    a = b = c = d = 0

    for i, j in combinations(range(n_samples), r=2):
        in_the_same_true = labels_true[i] == labels_true[j]
        in_the_same_pred = labels_pred[i] == labels_pred[j]

        a += 1 if in_the_same_true and in_the_same_pred else 0
        b += 1 if (not in_the_same_true) and (not in_the_same_pred) else 0
        c += 1 if in_the_same_true and (not in_the_same_pred) else 0
        d += 1 if (not in_the_same_true) and in_the_same_pred else 0

    denom = n_samples * (n_samples - 1) / 2
    r = (a + b) / denom
    return r
