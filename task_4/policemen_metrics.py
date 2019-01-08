import numpy as np
from PIL import Image
from kmeans import kmeans
from metrics.davies_bouldin import davies_bouldin_score
from metrics.calinski_harabaz import calinski_harabaz_score
from matplotlib import pyplot as plt


max_iter = 300
n_init = 1
clusters_n_range = list(range(2, 11))
image_name = "policemen"


davies_bouldin = []
calinski_harabaz = []

input_image_file = f"images/{image_name}.jpg"
image = np.array(Image.open(input_image_file))
X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))

for k in clusters_n_range:
    centroids, clustered = kmeans(X, n_clusters=k, max_iter=max_iter, n_init=n_init)
    davies_bouldin.append(davies_bouldin_score(X, clustered))
    calinski_harabaz.append(calinski_harabaz_score(X, clustered))

plt.subplot(211)
plt.title("Davies-Bouldin")
plt.plot(clusters_n_range, davies_bouldin)

plt.subplot(212)
plt.title("Calinski-Harabasz")
plt.plot(clusters_n_range, calinski_harabaz)

plt.show()