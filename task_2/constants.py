# Dict of parameters for GridSearchCV
svc_param_grid = [
        {
            'kernel': ['linear'],
            'C': [10 ** i for i in range(-4, 2)],
            "degree": [2, 3, 4]
        },
        {
            'kernel': ['poly', 'rbf', 'sigmoid'],
            'C': [10 ** i for i in range(-4, 2)],
            'gamma': [10 ** i for i in range(-4, 2)],
            "degree": [2, 3, 4]
        }
    ]

knn_param_grid = {
    'n_neighbors': [1, 2, 3, 5, 7, 10],
    'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
    'weights': ['uniform', 'distance'],
}

# Percentage of data, that is using for training
train_percentage = 80
