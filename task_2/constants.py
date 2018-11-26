# Dict of parameters for GridSearchCV
param_grid = [
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

# Percentage of data, that is using for training
train_percentage = 75
