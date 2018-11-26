import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from utils import plot_contours, set_clf_params, clf_accuracy, read_data
from constants import svc_param_grid


# Draw dots with the color depending on the cluster
def scatter_cluster_dots(plt, xx, yy):
    for i in range(len(xx)):
        plt.scatter(
            xx[i, 0], xx[i, 1],
            color="red" if yy[i] else "blue",
            zorder=3,
        )


# Read data from file
X, y, X_train, y_train, X_test, y_test = read_data('chips.txt')

# Create SVC classifier
clf = SVC()

# Calculate best parameters for SVC using GridSearchCV
set_clf_params(clf, svc_param_grid, X, y)

# Fit the model
clf.fit(X, y)

# Compute metrics
print(clf_accuracy(clf, X_train, y_train, X_test, y_test))

# Draw separating plane
x_min = X[:, 0].min()
x_max = X[:, 0].max()
y_min = X[:, 1].min()
y_max = X[:, 1].max()
h = 1e-3

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h)
)

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)


plt.subplot(121)
plt.title("Train features")
ax = plt.gca()
ax.set_facecolor('xkcd:salmon')
ax.set_facecolor((0.8, 0.8, 1))
plot_contours(plt, clf, xx, yy,
              colors=((0.8, 0.8, 1), (1, 0.8, 0.8)), levels=1)

scatter_cluster_dots(plt, X_train, y_train)


plt.subplot(122)
plt.title("Test features")
ax = plt.gca()
ax.set_facecolor('xkcd:salmon')
ax.set_facecolor((0.8, 0.8, 1))
plot_contours(plt, clf, xx, yy,
              colors=((0.8, 0.8, 1), (1, 0.8, 0.8)), levels=1)

scatter_cluster_dots(plt, X_test, y_test)
plt.show()
