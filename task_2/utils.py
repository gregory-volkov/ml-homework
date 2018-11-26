import numpy as np
from sklearn.model_selection import GridSearchCV


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
    clf.set_params(**best_params)
