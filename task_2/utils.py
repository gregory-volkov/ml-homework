import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from constants import *


# Function for drawing count
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, **params)
    ax.contour(xx, yy, Z, colors=((0, 0, 0, 0.2), ), linestyles="-", linewidths=0.3)


# Function for getting best hyperparameters
def set_clf_params(clf, param_grid, X, y):
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3, iid=False)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    print(f"best_params: {best_params}")
    clf.set_params(**best_params)


# Calculating of classifier performance
def clf_accuracy(clf, X_train, y_train, X_test, y_test):
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return (
        accuracy_score(y_train, pred_train),
        accuracy_score(y_test, pred_test)
    )


def read_data(filename):
    # Get and preprocess data
    data = np.genfromtxt(filename, delimiter=',')
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
    return X, y, X_train, y_train, X_test, y_test