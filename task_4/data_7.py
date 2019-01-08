import numpy as np
from kmeans import kmeans
from metrics.rand import rand_score
from metrics.fowlkes_mallows import fowlkes_mallows_score
from matplotlib import pyplot as plt


file_name = "task_2_data_7.txt"
dimension = 2

max_iter = 300
n_init = 1
clusters_n_range = list(range(2, 11))

rand = []
fow_mal = []

with open(file_name) as f:
    lines = f.read().splitlines()
    n_samples = len(lines)
    labels = np.empty((n_samples,), dtype=np.int8)
    samples = np.empty((n_samples, dimension), dtype=np.float32)

    for i, line in enumerate(lines):
        splitted = line.split()
        samples[i] = splitted[1:]
        labels[i] = splitted[0]

for k in clusters_n_range:
    centroids, clustered = kmeans(samples, n_clusters=k, max_iter=max_iter, n_init=n_init)
    rand.append(rand_score(clustered, labels))
    fow_mal.append(fowlkes_mallows_score(clustered, labels))


plt.subplot(211)
plt.title("Rand")
plt.plot(clusters_n_range, rand)

plt.subplot(212)
plt.title("Fowlkes-Mallows")
plt.plot(clusters_n_range, fow_mal)

plt.show()