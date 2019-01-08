import numpy as np


def kmeans(X, n_clusters, max_iter=300, n_init=10):
    def euclidean(a, b):
        return np.linalg.norm(a - b)

    n = X.shape[0]

    best_functional = -1
    best_centroids = np.zeros(shape=n, dtype=np.float64)
    best_clustered = np.zeros(shape=n, dtype=np.int8)

    for i_init in range(n_init):
        # Pick random points as cluster centers
        centroids = np.array(
            X[np.random.choice(range(n), size=n_clusters, replace=False), :],
            dtype=np.float32
        )
        # Put chosen points into cluster 0
        clustered = np.zeros(shape=n, dtype=np.int8)

        i_iter = 1  # iteration number
        converged = False
        while i_iter < max_iter and not converged:
            # Calculate cluster indices
            clustered = [np.argmin([euclidean(x, c) for c in centroids]) for x in X]
            # Update centroids
            converged = True
            for i in range(n_clusters):
                cluster = np.array([X[j] for j in range(n) if clustered[j] == i])
                new_centroid = np.mean(cluster, axis=0)
                if euclidean(centroids[i], new_centroid) > 0.5:
                    centroids[i] = new_centroid
                    converged = False
            i_iter += 1

        # Mitigate the local min problem: use the outputs with the least functional value
        target_functional = 0.
        for i in range(n_clusters):
            c = centroids[i]
            target_functional += sum([euclidean(c, X[j]) for j in range(n) if clustered[j] == i])
        if best_functional > -1:
            if target_functional < best_functional:
                best_functional = target_functional
                best_centroids = centroids
                best_clustered = clustered
        else:
            best_functional = target_functional
            best_centroids = centroids
            best_clustered = clustered

    return best_centroids, best_clustered
