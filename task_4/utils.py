import numpy as np


def normalize_labels(labels):
    cur_i = 0
    cur_label = min(labels)
    while True:
        labels = np.vectorize(lambda x: cur_i if x == cur_label else x)(labels)
        cur_i += 1
        try:
            cur_label = min(filter(lambda x: x >= cur_i, labels))
        except ValueError:
            break
    return labels
