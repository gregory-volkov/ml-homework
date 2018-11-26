import matplotlib.pyplot as plt
from utils import plot_contours, set_clf_params, clf_accuracy, read_data
from constants import knn_param_grid
from sklearn.neighbors import KNeighborsClassifier


# Function for marking incorrectly clustered items
def scatter_true_false_dots(plt, clf, xx, yy):
    y_pred = clf.predict(xx)
    for i in range(len(xx)):
        color = "red" if yy[i] else "blue"
        plt.scatter(
            xx[i, 0], xx[i, 1],
            color=color,
            zorder=3,
            marker=None if y_pred[i] == yy[i] else "X"
        )


# Read data from file
X, y, X_train, y_train, X_test, y_test = read_data('chips.txt')

# Create Knn classifier
clf = KNeighborsClassifier()


# Calculate best parameters for SVC using GridSearchCV
set_clf_params(clf, knn_param_grid, X, y)

# Fit the model
clf.fit(X, y)

# Compute metrics
print(clf_accuracy(clf, X_train, y_train, X_test, y_test))

# Incorrectly clustered points are marked as x
plt.subplot(121)
plt.title("Train features")
scatter_true_false_dots(plt, clf, X_train, y_train)

plt.subplot(122)
plt.title("Test features")
scatter_true_false_dots(plt, clf, X_test, y_test)

plt.show()
