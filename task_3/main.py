from PIL import Image
import numpy as np
from kmeans import kmeans

# Change this:
image_name = "peppers"


input_image_file = f"images/{image_name}.jpg"

image = np.array(Image.open(input_image_file))
X = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))

for k in range(2, 5):

    output_image_file = f"images/{image_name}_compressed_{k}.jpg"
    centroids, clustered = kmeans(X, n_clusters=k, max_iter=3, n_init=3)

    new_X = np.stack(
        centroids[i] for i in clustered
    )

    new_image = new_X.reshape(image.shape)
    Image.fromarray(new_image).save(output_image_file)
