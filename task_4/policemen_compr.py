import numpy as np
from PIL import Image
from kmeans import kmeans

max_iter = 300
n_init = 1
k = 5
image_name = "policemen"

input_image_file = f"images/{image_name}.jpg"
image = np.array(Image.open(input_image_file))
X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))

output_image_file = f"images/{image_name}_{k}_clusters.jpg"
centroids, clustered = kmeans(X, n_clusters=k, max_iter=max_iter, n_init=n_init)
centroids = centroids.astype(np.uint8)
new_X = np.stack(
    centroids[i] for i in clustered
)
new_image = new_X.reshape(image.shape)
Image.fromarray(new_image).save(output_image_file)
