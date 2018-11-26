import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from utils import plot_contours, set_svm_params


# Percentage of data, that is using for training
train_percentage = 75

# Get and preprocess data
data = np.genfromtxt('chips.txt', delimiter=',')
np.random.shuffle(data)
X, y = np.split(data, [-1], axis=1)

# Amount of data
samples_n = len(data)

# Splitting data for train and test
train_n = int(train_percentage / 100 * samples_n)
data_train = data[:train_n, :]
data_test = data[train_n:, :]

X_train, y_train = np.split(data_train, [-1], axis=1)
X_test, y_test = np.split(data_test, [-1], axis=1)

y, y_train, y_test = (y.ravel(), y_train.ravel(), y_test.ravel())

# Create SVC classifier
clf = SVC()

# Calculate best parameters for SVC using GridSearchCV
set_svm_params(clf, X, y)

# Fit the model
clf.fit(X, y)


# Print dots with the color depending on the cluster
def scatter_dots(xx, yy):
    for i in range(len(xx)):
        plt.scatter(
            xx[i, 0], xx[i, 1],
            color="red" if yy[i] else "blue",
            zorder=3
        )


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

scatter_dots(X_train, y_train)


plt.subplot(122)
plt.title("Test features")
ax = plt.gca()
ax.set_facecolor('xkcd:salmon')
ax.set_facecolor((0.8, 0.8, 1))
plot_contours(plt, clf, xx, yy,
              colors=((0.8, 0.8, 1), (1, 0.8, 0.8)), levels=1)

scatter_dots(X_test, y_test)
scatter_dots(X_test, y_test)
scatter_dots(X_test, y_test)
plt.show()
